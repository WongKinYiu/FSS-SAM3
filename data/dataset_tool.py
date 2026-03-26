import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO

COCO_ID_TO_NAME = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
    21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
    27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie',
    33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'
}

SORTED_COCO_IDS = sorted(list(COCO_ID_TO_NAME.keys()))

class COCOFSSDataset(Dataset):
    def __init__(self, lists_root, data_root, fold=0, k_shot=1, mode='val', seed=321):
        """
        Standard FSS Dataset with a dual-pool design.
        Supports K-shot evaluation by sampling K reference images per episode.
        """
        self.data_root = data_root
        self.k_shot = k_shot
        self.fold = fold
        self.mode = mode
        
        split = 'val2014' if mode == 'val' else 'train2014'
        self.img_dir = os.path.join(data_root, split)
        ann_file = os.path.join(data_root, 'annotations', f'instances_{split}.json')
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")
            
        self.coco = COCO(ann_file)
        
        # 1. Filter target classes for the current fold
        self.active_cat_ids = []
        for new_idx, old_id in enumerate(SORTED_COCO_IDS):
            if new_idx % 4 == fold:
                self.active_cat_ids.append(old_id)
                
        # 2. Build dual pools
        self.target_pool = {}
        self.support_pool = {}
        valid_cat_ids = []
        
        for cat_id in self.active_cat_ids:
            raw_img_ids = self.coco.getImgIds(catIds=cat_id)
            t_pool = []
            s_pool = []
            
            for img_id in raw_img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id)
                anns = self.coco.loadAnns(ann_ids)
                
                has_target = any(not a.get('iscrowd', 0) for a in anns)
                has_support = any(a['area'] > 1000 and not a.get('iscrowd', 0) for a in anns)
                
                if has_target: t_pool.append(img_id)
                if has_support: s_pool.append(img_id)
                    
            # Ensure enough data: at least 1 target and K support images
            if len(t_pool) > 0 and len(s_pool) >= self.k_shot + 1:
                self.target_pool[cat_id] = t_pool
                self.support_pool[cat_id] = s_pool
                valid_cat_ids.append(cat_id)
                
        self.active_cat_ids = valid_cat_ids
        
        # 3. Generate standard evaluation episodes
        self.test_items = []
        episode_num = 1000 if mode == 'val' else 4000 
        rng = random.Random(seed + fold)
        
        for _ in range(episode_num):
            cat_id = rng.choice(self.active_cat_ids)
            target_img_id = rng.choice(self.target_pool[cat_id])
            s_candidates = [x for x in self.support_pool[cat_id] if x != target_img_id]
            
            # Sample K_SHOT reference images instead of just 1
            if len(s_candidates) < self.k_shot:
                continue
            ref_img_ids = rng.sample(s_candidates, self.k_shot)
            
            self.test_items.append((cat_id, target_img_id, ref_img_ids))
                    
        print(f"Dataset Initialized: Fold {fold}, total {len(self.test_items)} episodes.")

    def __len__(self):
        return len(self.test_items)

    def __getitem__(self, idx):
        cat_id, target_img_id, ref_img_ids = self.test_items[idx]
        class_name = COCO_ID_TO_NAME[cat_id]
        
        target_data = self._load_target(target_img_id, cat_id)
        if target_data is None:
            return self.__getitem__((idx + 1) % len(self.test_items))
            
        # Load all K_SHOT reference images
        ref_datas = []
        for r_id in ref_img_ids:
            r_data = self._load_ref(r_id, cat_id)
            if r_data is None:
                break # If any reference fails, break to trigger retry
            ref_datas.append(r_data)
            
        # Ensure we successfully loaded exactly K_SHOT valid reference images
        if len(ref_datas) != self.k_shot:
            return self.__getitem__((idx + 1) % len(self.test_items))
            
        return ref_datas, [target_data], cat_id, class_name

    def _load_ref(self, img_id, cat_id):
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(path).convert("RGB")
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id)
        anns = self.coco.loadAnns(ann_ids)
        
        valid_anns = [a for a in anns if a['area'] > 1000 and not a.get('iscrowd', 0)]
        if not valid_anns: return None
        
        target_ann = random.choice(valid_anns)
        bbox = target_ann['bbox'] 
        return {'image': image, 'box': bbox, 'name': img_info['file_name']}

    def _load_target(self, img_id, cat_id):
        img_info = self.coco.loadImgs(img_id)[0]
        path = os.path.join(self.img_dir, img_info['file_name'])
        image = Image.open(path).convert("RGB")
        
        w, h = img_info['width'], img_info['height']
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_id)
        anns = self.coco.loadAnns(ann_ids)
        
        if not anns: return None
        for ann in anns:
            if ann.get('iscrowd', 0): continue
            m = self.coco.annToMask(ann)
            gt_mask = np.maximum(gt_mask, m)
            
        if gt_mask.max() == 0: return None
        return {'image': image, 'mask': gt_mask, 'name': img_info['file_name']}
    

PASCAL_ID_TO_NAME = {
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle',
    6: 'bus', 7: 'car', 8: 'cat', 9: 'chair', 10: 'cow',
    11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person',
    16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'
}

class PascalVOCFSSDataset(Dataset):
    def __init__(self, data_root, fold=0, k_shot=1, mode='val', seed=321):
        self.data_root = data_root
        self.k_shot = k_shot
        self.fold = fold
        self.mode = mode
        
        self.img_dir = os.path.join(data_root, 'JPEGImages')
        self.mask_dir = os.path.join(data_root, 'SegmentationClass')
        self.obj_dir = os.path.join(data_root, 'SegmentationObject') # for support image
        split_file = os.path.join(data_root, 'ImageSets', 'Segmentation', f'{mode}.txt')
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"PASCAL split file not found: {split_file}")
            
        with open(split_file, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
            
        start_id = fold * 5 + 1
        end_id = start_id + 5
        self.active_cat_ids = list(range(start_id, end_id))
        
        print(f"Loading PASCAL VOC ({mode}) for Fold {fold}")
        
        self.target_pool = {cat_id: [] for cat_id in self.active_cat_ids}
        self.support_pool = {cat_id: [] for cat_id in self.active_cat_ids}
        
        for file_name in self.file_names:
            mask_path = os.path.join(self.mask_dir, file_name + '.png')
            if not os.path.exists(mask_path): continue
                
            mask = np.array(Image.open(mask_path))
            
            for cat_id in self.active_cat_ids:
                cat_mask = (mask == cat_id)
                if not np.any(cat_mask): continue
                
                ys, xs = np.where(cat_mask)
                area = len(xs) 
                
                self.target_pool[cat_id].append(file_name)
                if area > 1000:
                    self.support_pool[cat_id].append(file_name)
                    
        self.active_cat_ids = [cid for cid in self.active_cat_ids 
                               if len(self.target_pool[cid]) > 0 and len(self.support_pool[cid]) >= self.k_shot + 1]
                               
        self.test_items = []
        episode_num = 1000 
        rng = random.Random(seed + fold)
        
        for _ in range(episode_num):
            cat_id = rng.choice(self.active_cat_ids)
            target_file = rng.choice(self.target_pool[cat_id])
            s_candidates = [x for x in self.support_pool[cat_id] if x != target_file]
            
            if len(s_candidates) < self.k_shot: continue
            ref_files = rng.sample(s_candidates, self.k_shot)
            
            self.test_items.append((cat_id, target_file, ref_files))
            
        print(f"Dataset Initialized: Fold {fold}, total {len(self.test_items)} episodes.")

    def __len__(self):
        return len(self.test_items)

    def __getitem__(self, idx):
        cat_id, target_file, ref_files = self.test_items[idx]
        class_name = PASCAL_ID_TO_NAME[cat_id]
        
        target_data = self._load_data(target_file, cat_id, is_ref=False)
        if target_data is None:
            return self.__getitem__((idx + 1) % len(self.test_items))
            
        ref_datas = []
        for r_file in ref_files:
            r_data = self._load_data(r_file, cat_id, is_ref=True)
            if r_data is None: break
            ref_datas.append(r_data)
            
        if len(ref_datas) != self.k_shot:
            return self.__getitem__((idx + 1) % len(self.test_items))
            
        return ref_datas, [target_data], cat_id, class_name

    def _load_data(self, file_name, cat_id, is_ref):
        img_path = os.path.join(self.img_dir, file_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, file_name + '.png')
        
        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        semantic_mask = (mask == cat_id)
        
        if not np.any(semantic_mask): return None

        if is_ref:
            bbox = None
            obj_path = os.path.join(self.obj_dir, file_name + '.png')
            if os.path.exists(obj_path):
                obj_mask = np.array(Image.open(obj_path))
                valid_pixels = obj_mask[semantic_mask]
                unique_ids = [uid for uid in np.unique(valid_pixels) if uid != 0 and uid != 255]
                
                if unique_ids:
                    target_uid = max(unique_ids, key=lambda uid: np.sum(obj_mask == uid))
                    instance_mask = (obj_mask == target_uid) & semantic_mask
                    
                    ys, xs = np.where(instance_mask)
                    if len(xs) > 0:
                        img_w, img_h = image.size
                        x_min = max(0, xs.min())
                        y_min = max(0, ys.min())
                        x_max = min(img_w - 1, xs.max())
                        y_max = min(img_h - 1, ys.max())
                        
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            
            if bbox is None: return None
            return {'image': image, 'box': bbox, 'name': file_name}
        else:
            return {'image': image, 'mask': semantic_mask.astype(np.uint8), 'name': file_name}