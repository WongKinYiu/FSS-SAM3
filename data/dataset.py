import os
import cv2
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

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

class COCOFSSDataset(Dataset):
    def __init__(self, lists_root, data_root, fold=0, k_shot=1, mode='val'):
        """
        Args:
            lists_root: Path to the lists folder (e.g., ~/fsssam3/data/lists/coco/fss_list)
            data_root: Root directory of the dataset (e.g., ~/fsssam3/data)
            fold: 0, 1, 2, 3
            k_shot: Number of support references (currently 1)
            mode: 'train' or 'val'
        """
        self.data_root = data_root
        self.k_shot = k_shot
        
        # Define paths
        list_dir = os.path.join(lists_root, mode)
        data_list_path = os.path.join(list_dir, f'data_list_{fold}.txt')
        sub_class_path = os.path.join(list_dir, f'sub_class_file_list_{fold}.txt')

        # 1. Read query list and fix relative paths
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
        
        self.data_list = []
        for line in lines:
            img_p, ann_p = line.strip().replace('../data', data_root).split(' ')
            self.data_list.append((img_p, ann_p))

        # 2. Read support dictionary and fix relative paths
        with open(sub_class_path, 'r') as f:
            dict_content = f.read().replace('../data', data_root)
            self.sub_class_dict = eval(dict_content)

        print(f"Dataset Initialized: Fold {fold}, Mode {mode}, {len(self.data_list)} items.")

    def __len__(self):
        return len(self.data_list)

    def _get_bbox(self, mask):
        """Extract a single instance bbox from a binary mask."""
        mask_uint8 = mask.astype(np.uint8)
        
        # Find all connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        valid_boxes = []
        # Skip background (label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter out noise (area < 1000)
            if area > 1000:
                x = float(stats[i, cv2.CC_STAT_LEFT])
                y = float(stats[i, cv2.CC_STAT_TOP])
                w = float(stats[i, cv2.CC_STAT_WIDTH])
                h = float(stats[i, cv2.CC_STAT_HEIGHT])
                valid_boxes.append([x, y, w, h])
                
        # Randomly select one valid box
        if len(valid_boxes) > 0:
            return random.choice(valid_boxes)
            
        # --- Fallback mechanisms ---
        # Fallback: find the largest component if none are > 1000
        if num_labels > 1:
            max_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
            x = float(stats[max_label, cv2.CC_STAT_LEFT])
            y = float(stats[max_label, cv2.CC_STAT_TOP])
            w = float(stats[max_label, cv2.CC_STAT_WIDTH])
            h = float(stats[max_label, cv2.CC_STAT_HEIGHT])
            return [x, y, w, h]
            
        # Fallback: empty box if no foreground
        return [0.0, 0.0, 0.0, 0.0]

    def __getitem__(self, idx):
        # --- Process Query (Target) ---
        query_img_p, query_ann_p = self.data_list[idx]
        
        # Read query annotation to determine categories
        raw_ann = np.array(Image.open(query_ann_p))
        present_cats = np.unique(raw_ann)
        present_cats = present_cats[(present_cats > 0) & (present_cats < 255)]
        
        # Keep only categories present in the current fold
        valid_cats = [c for c in present_cats if c in self.sub_class_dict]
        
        # If no valid categories, pick another random image
        if len(valid_cats) == 0: 
            return self.__getitem__(random.randint(0, len(self.data_list)-1))
            
        # Randomly select a target category
        cat_id = int(random.choice(valid_cats))
        class_name = COCO_ID_TO_NAME.get(cat_id, f"ID_{cat_id}")

        query_img = Image.open(query_img_p).convert("RGB")
        query_mask = (raw_ann == cat_id).astype(np.uint8)

        # --- Process Support (Reference) ---
        support_pool = self.sub_class_dict[cat_id]
        
        # Exclude the query image from support candidates
        candidates = [p for p in support_pool if p[0] != query_img_p]
        
        # If not enough candidates, pick another random image
        if len(candidates) < self.k_shot:
            return self.__getitem__(random.randint(0, len(self.data_list)-1))
        
        # Randomly sample k shots
        selected_supports = random.sample(candidates, self.k_shot)
        
        support_datas = []
        for s_img_p, s_ann_p in selected_supports:
            s_img = Image.open(s_img_p).convert("RGB")
            s_mask_full = np.array(Image.open(s_ann_p))
            s_mask_bin = (s_mask_full == cat_id).astype(np.uint8)
            
            support_datas.append({
                'image': s_img,
                'box': self._get_bbox(s_mask_bin),
                'name': os.path.basename(s_img_p)
            })

        # Return format for the Evaluator
        ref_data = support_datas[0]
        target_data = {'image': query_img, 'mask': query_mask, 'name': os.path.basename(query_img_p)}
        
        return ref_data, [target_data], cat_id, class_name