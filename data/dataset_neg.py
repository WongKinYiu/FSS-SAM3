import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from collections import defaultdict
import random
import itertools

random.seed(42)

class CooccurrenceDataset(Dataset):
    def __init__(self, ann_file, max_samples_per_cat=50, max_samples_per_pair=3, num_shots=5):
        self.coco = COCO(ann_file)
        self.num_shots = num_shots
        self.cat_id_to_name = {c['id']: c['name'] for c in self.coco.loadCats(self.coco.getCatIds())}
        
        # 1. Build Index
        self.img_to_anns = defaultdict(list)
        for ann in self.coco.dataset['annotations']:
            if not ann['iscrowd'] and ann['area'] > 1000:
                self.img_to_anns[ann['image_id']].append(ann)

        self.pair_to_images = defaultdict(list)
        for img_id, anns in self.img_to_anns.items():
            current_cats = sorted(list(set([a['category_id'] for a in anns])))
            for i, j in itertools.combinations(current_cats, 2):
                self.pair_to_images[(i, j)].append(img_id)

        # 2. Build Tasks
        self.tasks = []
        pos_cat_counts = defaultdict(int)
        all_cat_ids = sorted(self.cat_id_to_name.keys())
        all_possible_pairs = list(itertools.permutations(all_cat_ids, 2))
        random.shuffle(all_possible_pairs)

        for cat_pos, cat_neg in all_possible_pairs:
            if pos_cat_counts[cat_pos] >= max_samples_per_cat:
                continue
            
            pair_key = tuple(sorted((cat_pos, cat_neg)))
            candidate_imgs = self.pair_to_images.get(pair_key, [])
            if len(candidate_imgs) < (1 + self.num_shots):
                continue
            
            num_samples = min(len(candidate_imgs), max_samples_per_pair)
            num_samples = min(num_samples, max_samples_per_cat - pos_cat_counts[cat_pos])
            
            # Global random sample for targets
            selected_targets = random.sample(candidate_imgs, num_samples)
            for tgt_id in selected_targets:
                self.tasks.append({
                    'pos_cat': cat_pos,
                    'neg_cat': cat_neg,
                    'target_img_id': tgt_id,
                    'all_candidates': candidate_imgs
                })
                pos_cat_counts[cat_pos] += 1

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        cat_pos, cat_neg, tgt_id = task['pos_cat'], task['neg_cat'], task['target_img_id']
        
        # Get target annotations
        t_anns = self.img_to_anns[tgt_id]
        t_pos = max([a for a in t_anns if a['category_id'] == cat_pos], key=lambda x: x['area'])
        t_neg = max([a for a in t_anns if a['category_id'] == cat_neg], key=lambda x: x['area'])

        # Randomly sample refs on-the-fly for diversity
        ref_candidates = [i for i in task['all_candidates'] if i != tgt_id]
        selected_ref_ids = random.sample(ref_candidates, self.num_shots)

        refs = []
        for rid in selected_ref_ids:
            r_anns = self.img_to_anns[rid]
            r_p = max([a for a in r_anns if a['category_id'] == cat_pos], key=lambda x: x['area'])
            r_n = max([a for a in r_anns if a['category_id'] == cat_neg], key=lambda x: x['area'])
            refs.append({"image_id": rid, "pos_box": r_p['bbox'], "neg_box": r_n['bbox']})

        # Just return metadata, images will be loaded in the evaluator
        return {
            "pos_category_id": cat_pos,
            "neg_category_id": cat_neg,
            "pos_cat_name": self.cat_id_to_name[cat_pos],
            "neg_cat_name": self.cat_id_to_name[cat_neg],
            "target": {"image_id": tgt_id, "pos_box": t_pos['bbox'], "neg_box": t_neg['bbox']},
            "refs": refs
        }