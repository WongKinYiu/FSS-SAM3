# =========================================================
# fsod_eval.py
# 1-shot / 5-shot
# =========================================================
import os
import json
import torch
import random
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from matplotlib.patches import Rectangle

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from sam3.model.sam3_image_processor import Sam3Processor
from sam3 import build_sam3_image_model

from data.fsod_pascal_dataset import FSODPascalDataset
from data.fsod_coco_dataset import FSODCOCODataset

# =========================================================
# ROOT
# =========================================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")

# =========================================================
# CONFIG
# =========================================================
SAVE_VIS = True
SAVE_VIS_NUM = 50
SAVE_DIR = os.path.join(ROOT_DIR, "fsod_vis")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------
# Pascal
# ---------------------------------------------------------
PASCAL_GT_JSON = os.path.join(DATA_DIR, "PascalVOC", "VOC2007Test", "voc07test_coco_format.json")
PASCAL_IMAGE_ROOT = DATASET_DIR

# ---------------------------------------------------------
# COCO
# ---------------------------------------------------------
COCO_GT_JSON = os.path.join(DATA_DIR, "coco", "annotations", "instances_val2017.json")
COCO_SUPPORT_JSON = os.path.join(DATA_DIR, "coco", "few_shot_10shot_seed33.json")

# ---------------------------------------------------------
# Pascal splits
# ---------------------------------------------------------
PASCAL_SPLITS = {
    "split1": {
        1: os.path.join(DATA_DIR, "PascalVOC", "vocsplit", "split1", "1shot_seed33.json"),
        5: os.path.join(DATA_DIR, "PascalVOC", "vocsplit", "split1", "5shot_seed33.json"),
        "categories": ["bus", "sofa", "cow", "bird", "motorbike"]
    },
    "split2": {
        1: os.path.join(DATA_DIR, "PascalVOC", "vocsplit", "split2", "1shot_seed33.json"),
        5: os.path.join(DATA_DIR, "PascalVOC", "vocsplit", "split2", "5shot_seed33.json"),
        "categories": ["horse", "aeroplane", "bottle", "sofa", "cow"]
    },
    "split3": {
        1: os.path.join(DATA_DIR, "PascalVOC", "vocsplit", "split3", "1shot_seed33.json"),
        5: os.path.join(DATA_DIR, "PascalVOC", "vocsplit", "split3", "5shot_seed33.json"),
        "categories": ["cat", "motorbike", "boat", "sofa", "sheep"]
    }
}

# =========================================================
# ARGS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pascal", choices=["pascal", "coco"])
    parser.add_argument("--shot", type=int, default=1, choices=[1, 5])
    return parser.parse_args()

# =========================================================
# UTILS
# =========================================================
def setup_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =========================================================
# EVALUATOR
# =========================================================
class FSODEvaluator:
    def __init__(self, canvas_size=1008):
        self.canvas_size = canvas_size
        bpe_path = os.path.join(ROOT_DIR, "assets", "bpe_simple_vocab_16e6.txt.gz")
        model = build_sam3_image_model(bpe_path=bpe_path)
        self.processor = Sam3Processor(model)
        self.device = self.processor.device
        print("SAM3 loaded")

    # =====================================================
    # 1-shot canvas
    # =====================================================
    def create_single_canvas(self, support_data, query_image):
        support_image = support_data["image"]
        canvas = Image.new('RGB', (self.canvas_size, self.canvas_size), (0, 0, 0))
        split_ratio = 0.6
        
        support_h = int(self.canvas_size * split_ratio)
        query_h = (self.canvas_size - support_h)
        
        support_resized = support_image.resize((self.canvas_size, support_h), Image.BILINEAR)
        query_resized = query_image.resize((self.canvas_size, query_h), Image.BILINEAR)

        canvas.paste(support_resized, (0, 0))
        canvas.paste(query_resized, (0, support_h))

        bx, by, bw, bh = support_data["box"]
        img_w, img_h = support_image.size

        scale_x = self.canvas_size / img_w
        scale_y = support_h / img_h

        scaled_box = [
            bx * scale_x,
            by * scale_y,
            bw * scale_x,
            bh * scale_y
        ]

        placements = [
            {
                "type": "support",
                "offset": (0, 0),
                "size": (self.canvas_size, support_h),
                "orig_size": (img_w, img_h),
                "scaled_box": scaled_box
            },
            {
                "type": "query",
                "offset": (0, support_h),
                "size": (self.canvas_size, query_h),
                "orig_size": query_image.size
            }
        ]

        return canvas, placements

    # =====================================================
    # multi-shot canvas
    # =====================================================
    def create_multi_canvas(self, support_datas, query_image):
        
        assert len(support_datas) <= 5, \
            "Current layout only supports <=5-shot"
        
        canvas = Image.new('RGB', (self.canvas_size, self.canvas_size), (0, 0, 0))
        placements = []
        split_ratio = 0.33

        support_thickness = int(self.canvas_size * split_ratio)
        target_size = self.canvas_size - support_thickness

        top_w = self.canvas_size // 3
        left_h = target_size // 2

        layouts = [
            (0, 0, top_w, support_thickness),
            (top_w, 0, top_w, support_thickness),
            (top_w * 2, 0, self.canvas_size - top_w * 2, support_thickness),
            (0, support_thickness, support_thickness, left_h),
            (0, support_thickness + left_h, support_thickness, target_size - left_h),
            (support_thickness, support_thickness, target_size, target_size)
        ]

        # -------------------------------------------------
        # supports
        # -------------------------------------------------
        for idx, support_data in enumerate(support_datas):
            img = support_data["image"]
            ox, oy, cw, ch = layouts[idx]
            img_w, img_h = img.size
            img_resized = img.resize((cw, ch), Image.BILINEAR)
            canvas.paste(img_resized, (ox, oy))

            bx, by, bw, bh = support_data["box"]

            scale_x = cw / img_w
            scale_y = ch / img_h

            scaled_box = [
                ox + bx * scale_x,
                oy + by * scale_y,
                bw * scale_x,
                bh * scale_y
            ]

            placements.append({
                "type": "support",
                "offset": (ox, oy),
                "size": (cw, ch),
                "orig_size": (img_w, img_h),
                "scaled_box": scaled_box
            })

        # -------------------------------------------------
        # query
        # -------------------------------------------------
        qx, qy, qw, qh = layouts[5]
        query_resized = query_image.resize((qw, qh), Image.BILINEAR)
        canvas.paste(query_resized, (qx, qy))

        placements.append({
            "type": "query",
            "offset": (qx, qy),
            "size": (qw, qh),
            "orig_size": query_image.size
        })

        return canvas, placements

    # =====================================================
    # convert back
    # =====================================================
    def canvas_box_to_query_box(self, box, query_info):
        qx, qy = query_info["offset"]
        query_w, query_h = query_info["size"]
        orig_w, orig_h = query_info["orig_size"]

        scale_x = orig_w / query_w
        scale_y = orig_h / query_h

        x1, y1, x2, y2 = box

        if x2 <= qx or y2 <= qy:
            return None

        if x1 >= qx + query_w or y1 >= qy + query_h:
            return None

        x1, x2 = max(0, x1 - qx), min(query_w, x2 - qx)
        y1, y2 = max(0, y1 - qy), min(query_h, y2 - qy)

        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y

        w, h = x2 - x1, y2 - y1

        if w <= 1 or h <= 1:
            return None

        return [float(x1), float(y1), float(w), float(h)]

    # =====================================================
    # inference
    # =====================================================
    def run_inference(self, support_datas, query_image, cls_name):
        # -------------------------------------------------
        # layout
        # -------------------------------------------------
        if len(support_datas) == 1:
            canvas, placements = self.create_single_canvas(support_datas[0], query_image)
        else:
            canvas, placements = self.create_multi_canvas(support_datas, query_image)

        state = self.processor.set_image(canvas)
        self.processor.reset_all_prompts(state)

        # -------------------------------------------------
        # prompts
        # -------------------------------------------------
        for placement in placements:
            if placement["type"] != "support":
                continue

            sbx, sby, sbw, sbh = placement["scaled_box"]
            cx = (sbx + sbw / 2) / self.canvas_size
            cy = (sby + sbh / 2) / self.canvas_size
            nw = sbw / self.canvas_size
            nh = sbh / self.canvas_size

            norm_box = [cx, cy, nw, nh]
            state = self.processor.add_geometric_prompt(state=state, box=norm_box, label=True)

        pred_boxes = state["boxes"]
        pred_scores = state["scores"]

        if pred_boxes is None:
            return [], []

        pred_boxes = pred_boxes.detach().cpu().numpy()
        pred_scores = pred_scores.detach().cpu().numpy()

        final_boxes = []
        final_scores = []
        query_info = next(p for p in placements if p["type"] == "query")

        for box, score in zip(pred_boxes, pred_scores):
            query_box = self.canvas_box_to_query_box(box, query_info)
            if query_box is not None:
                final_boxes.append(query_box)
                final_scores.append(float(score))

        return final_boxes, final_scores


# =========================================================
# VIS
# =========================================================
def visualize_episode(support_datas, query_image, gt_boxes, pred_boxes, pred_scores, cls_name, image_id, save_path):
    
    num_supports = len(support_datas)
    
    if num_supports == 1:
        fig = plt.figure(figsize=(10, 12))
    else:
        fig = plt.figure(figsize=(14, 14))

    # -----------------------------------------------------
    # supports
    # -----------------------------------------------------
    for i, support_data in enumerate(support_datas):
        ax = plt.subplot(3, 2, i + 1)
        img = np.array(support_data["image"])
        ax.imshow(img)
        x, y, w, h = support_data["box"]
        rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Support {i+1}")
        ax.axis("off")

    # -----------------------------------------------------
    # query
    # -----------------------------------------------------
    query_subplot_idx = 2 if num_supports == 1 else 6
    axq = plt.subplot(3, 2, query_subplot_idx)
    query_np = np.array(query_image)
    axq.imshow(query_np)

    for gt_box in gt_boxes:
        x, y, w, h = gt_box
        rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='lime', facecolor='none')
        axq.add_patch(rect)

    for pred_box, score in zip(pred_boxes, pred_scores):
        x, y, w, h = pred_box
        rect = Rectangle((x, y), w, h, linewidth=3, edgecolor='red', facecolor='none')
        axq.add_patch(rect)
        axq.text(x, y - 5, f"{score:.2f}", fontsize=10, color='white', bbox=dict(facecolor='red'))

    axq.set_title(f"Query | GT={len(gt_boxes)} | Pred={len(pred_boxes)}")
    axq.axis("off")

    plt.suptitle(f"{cls_name} | image_id={image_id}", fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()


# =========================================================
# COCO EVAL
# =========================================================
def evaluate_predictions(gt_json, pred_json, target_categories):
    coco_gt = COCO(gt_json)
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []

    coco_pred = coco_gt.loadRes(pred_json)
    coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
    
    target_cat_ids = coco_gt.getCatIds(catNms=target_categories)
    coco_eval.params.catIds = target_cat_ids
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats


# =========================================================
# RUN
# =========================================================
def run_dataset(dataset, evaluator, gt_json, target_categories, dataset_name, shot):
    all_predictions = []
    seen = set()

    for idx in tqdm(range(len(dataset))):
        # -------------------------------------------------
        # Pascal
        # -------------------------------------------------
        if dataset_name == "pascal":
            ref_data, target_data, cat_id, cls_name = dataset[idx]
            if shot == 1:
                support_datas = [{"image": ref_data["image"], "box": ref_data["box"]}]
            else:
                support_datas = ref_data

            query_image = target_data["image"]
            gt_boxes = target_data["gt_boxes"]
            image_id = target_data["image_id"]

        # -------------------------------------------------
        # COCO
        # -------------------------------------------------
        elif dataset_name == "coco":
            sample = dataset[idx]
            support_datas = []
            for s_img, s_box in zip(sample["support_images"], sample["support_boxes"]):
                support_datas.append({"image": s_img, "box": s_box})

            query_image = sample["query_image"]
            gt_boxes = sample["query_boxes"]
            image_id = sample["query_image_id"]
            cat_id = sample["target_class"]
            cls_name = dataset.cat_id_to_name[cat_id]
        else:
            raise ValueError(dataset_name)

        key = (image_id, cat_id)
        if key in seen:
            continue
        seen.add(key)

        pred_boxes, pred_scores = evaluator.run_inference(support_datas, query_image, cls_name)

        # -------------------------------------------------
        # visualization
        # -------------------------------------------------
        if SAVE_VIS and idx < SAVE_VIS_NUM:
            save_path = os.path.join(SAVE_DIR, f"{dataset_name}_{shot}shot_{idx}_{cls_name}.jpg")
            visualize_episode(support_datas, query_image, gt_boxes, pred_boxes, pred_scores, cls_name, image_id, save_path)

        # -------------------------------------------------
        # predictions
        # -------------------------------------------------
        for pred_box, score in zip(pred_boxes, pred_scores):
            x, y, w, h = pred_box
            pred = {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "bbox": [float(x), float(y), float(w), float(h)],
                "score": float(score)
            }
            all_predictions.append(pred)

    pred_json_path = f"./{dataset_name}_{shot}shot_predictions.json"
    with open(pred_json_path, "w") as f:
        json.dump(all_predictions, f)
    print(f"\nSaved: {pred_json_path}")

    stats = evaluate_predictions(gt_json, pred_json_path, target_categories)
    return stats


# =========================================================
# MAIN
# =========================================================
def main():
    args = parse_args()
    setup_seed(123)
    evaluator = FSODEvaluator(canvas_size=1008)

    # =====================================================
    # Pascal
    # =====================================================
    if args.dataset == "pascal":
        all_results = {}
        for split_name, split_cfg in PASCAL_SPLITS.items():
            print("\n=================================================")
            print(f" RUNNING {split_name.upper()} {args.shot}-SHOT")
            print("=================================================\n")

            dataset = FSODPascalDataset(
                support_json=split_cfg[args.shot],
                query_json=PASCAL_GT_JSON,
                image_root=PASCAL_IMAGE_ROOT,
                target_categories=split_cfg["categories"]
            )

            stats = run_dataset(
                dataset=dataset,
                evaluator=evaluator,
                gt_json=PASCAL_GT_JSON,
                target_categories=split_cfg["categories"],
                dataset_name="pascal",
                shot=args.shot
            )
            all_results[split_name] = stats

        # -------------------------------------------------
        # summary
        # -------------------------------------------------
        print("\n=================================================")
        print(f" FINAL {args.shot}-SHOT RESULTS")
        print("=================================================\n")

        mean_ap50 = 0.0
        mean_ap = 0.0
        for split_name in PASCAL_SPLITS:
            stats = all_results[split_name]
            ap, ap50 = stats[0], stats[1]
            mean_ap += ap
            mean_ap50 += ap50
            print(f"{split_name}: AP50={ap50:.4f} | AP={ap:.4f}")

        mean_ap /= len(PASCAL_SPLITS)
        mean_ap50 /= len(PASCAL_SPLITS)

        print("\n-------------------------------------------------")
        print(f"Mean AP50 : {mean_ap50:.4f}")
        print(f"Mean AP   : {mean_ap:.4f}")

    # =====================================================
    # COCO
    # =====================================================
    elif args.dataset == "coco":
        print("\n=================================================")
        print(f" RUNNING COCO {args.shot}-SHOT")
        print("=================================================\n")

        dataset = FSODCOCODataset(
            ann_root=os.path.join(DATA_DIR, "coco", "annotations"),
            img_root=os.path.join(DATASET_DIR, "coco"),
            shot=args.shot,
            fixed_support_json=COCO_SUPPORT_JSON,
            fixed_support_shot=args.shot,
        )

        stats = run_dataset(
            dataset=dataset,
            evaluator=evaluator,
            gt_json=COCO_GT_JSON,
            target_categories=dataset.FSOD_COCO_CATEGORIES,
            dataset_name="coco",
            shot=args.shot
        )

        print("\n=================================================")
        print(f" COCO {args.shot}-SHOT RESULTS")
        print("=================================================\n")
        print(f"AP   : {stats[0]:.4f}")
        print(f"AP50 : {stats[1]:.4f}")

    print("\n=================================================\n")


if __name__ == "__main__":
    main()