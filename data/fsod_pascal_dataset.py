import os
import json

from PIL import Image
from torch.utils.data import Dataset


class FSODPascalDataset(Dataset):

    def __init__(
        self,
        support_json,
        query_json,
        image_root,
        target_categories
    ):

        self.image_root = image_root
        self.target_categories = target_categories

        # -------------------------
        # load support json
        # -------------------------

        with open(support_json, 'r') as f:
            self.support_data = json.load(f)

        # -------------------------
        # load query json
        # -------------------------

        with open(query_json, 'r') as f:
            self.query_data = json.load(f)

        self.images = self.query_data['images']
        self.annotations = self.query_data['annotations']
        self.categories = self.query_data['categories']

        # -------------------------
        # category mapping
        # -------------------------

        self.name2id = {}
        self.id2name = {}

        for cat in self.categories:

            self.name2id[cat['name']] = cat['id']
            self.id2name[cat['id']] = cat['name']

        # -------------------------
        # image_id -> image_info
        # -------------------------

        self.imageid2info = {}

        for img in self.images:

            self.imageid2info[img['id']] = img

        # -------------------------
        # category_id -> annotations
        # -------------------------

        self.catid2anns = {}

        for ann in self.annotations:

            cat_id = ann['category_id']

            if cat_id not in self.catid2anns:
                self.catid2anns[cat_id] = []

            self.catid2anns[cat_id].append(ann)

        # -------------------------
        # build episodes
        # -------------------------

        self.episodes = []

        for cls_name in target_categories:

            if cls_name not in self.name2id:
                print(f"[WARNING] {cls_name} not found")
                continue

            cat_id = self.name2id[cls_name]

            if cat_id not in self.catid2anns:
                print(f"[WARNING] No annotations for {cls_name}")
                continue

            anns = self.catid2anns[cat_id]

            for ann in anns:

                self.episodes.append({
                    'cls_name': cls_name,
                    'cat_id': cat_id,
                    'ann': ann
                })

        print(f"Total Episodes: {len(self.episodes)}")


    def __len__(self):
        return len(self.episodes)


    def __getitem__(self, idx):

        episode = self.episodes[idx]

        cls_name = episode['cls_name']
        cat_id = episode['cat_id']

        # =========================================================
        # SUPPORT
        # =========================================================

        support_items = self.support_data[cls_name]

        ref_data = []

        for support_item in support_items:

            support_img_path = os.path.join(
                self.image_root,
                support_item['image']
            )

            support_image = Image.open(
                support_img_path
            ).convert('RGB')

            support_box = support_item['bbox']

            ref_data.append({
                'image': support_image,
                'box': support_box
            })

        # =========================================================
        # QUERY
        # =========================================================

        ann = episode['ann']

        image_id = ann['image_id']

        img_info = self.imageid2info[image_id]

        query_img_path = os.path.join(
            self.image_root,
            "PascalVOC",
            "VOC2007Test",
            "VOC2007",
            "JPEGImages",
            img_info['file_name']
        )

        query_image = Image.open(
            query_img_path
        ).convert('RGB')

        # ---------------------------------------------------------
        # gather all same-category GT boxes
        # ---------------------------------------------------------

        all_gt_boxes = []

        for a in self.catid2anns[cat_id]:

            if a['image_id'] == image_id:

                all_gt_boxes.append(a['bbox'])

        target_data = {
            'image': query_image,
            'gt_boxes': all_gt_boxes,
            'image_id': image_id
        }

        return ref_data, target_data, cat_id, cls_name