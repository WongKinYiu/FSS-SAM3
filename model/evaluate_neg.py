# treat support inference score < 0.5 as negative samples
import torch
import numpy as np
from PIL import Image
import os
import sys
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects 
import torch.nn.functional as F
from typing import List, Dict
import torchvision.ops as ops

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import SAM3
try:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3 import build_sam3_image_model
    from data.dataset_tool import COCOFSSDataset, COCO_ID_TO_NAME
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

@torch.inference_mode()
def add_multiple_geometric_prompts(processor: Sam3Processor, boxes: List[List[float]], labels: List[bool], state: Dict):
    """
    boxes: list of [center_x, center_y, width, height] (normalized)
    labels: list of boolean (True for positive, False for negative)
    """
    if "backbone_out" not in state:
        raise ValueError("You must call set_image before set_text_prompt")
        
    if "language_features" not in state["backbone_out"]:
        dummy_text_outputs = processor.model.backbone.forward_text(
            ["visual"], device=processor.device
        )
        state["backbone_out"].update(dummy_text_outputs)

    if "geometric_prompt" not in state:
        state["geometric_prompt"] = processor.model._get_dummy_prompt()

    if len(boxes) == 0:
        return processor._forward_grounding(state)

    for box, label in zip(boxes, labels):
        box_tensor = torch.tensor(box, device=processor.device, dtype=torch.float32).view(1, 1, 4)
        label_tensor = torch.tensor([label], device=processor.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(box_tensor, label_tensor)

    return processor._forward_grounding(state)

# Metric
class ClassWiseIOUMetric:
    def __init__(self):
        self.intersections = {}
        self.unions = {}
        
    def update(self, pred, gt, cat_id):
        pred_bin = (pred > 0)
        gt_bin = (gt > 0)
        inter_val = np.logical_and(pred_bin, gt_bin).sum()
        union_val = np.logical_or(pred_bin, gt_bin).sum()
        
        if cat_id not in self.intersections:
            self.intersections[cat_id] = 0.0
            self.unions[cat_id] = 0.0
            
        self.intersections[cat_id] += inter_val
        self.unions[cat_id] += union_val
        
    def compute(self):
        ious = []
        for cat_id in self.intersections:
            total_inter = self.intersections[cat_id]
            total_union = self.unions[cat_id]
            if total_union > 0:
                ious.append(total_inter / total_union)
            else:
                ious.append(0.0)
        return np.mean(ious) if ious else 0.0
    
class FBIouMetric:
    def __init__(self):
        self.fg_inter = 0.0
        self.fg_union = 0.0
        self.bg_inter = 0.0
        self.bg_union = 0.0
        
    def update(self, pred, gt):
        pred_fg = (pred > 0)
        gt_fg = (gt > 0)
        pred_bg = ~pred_fg
        gt_bg = ~gt_fg
        
        self.fg_inter += np.logical_and(pred_fg, gt_fg).sum()
        self.fg_union += np.logical_or(pred_fg, gt_fg).sum()
        
        self.bg_inter += np.logical_and(pred_bg, gt_bg).sum()
        self.bg_union += np.logical_or(pred_bg, gt_bg).sum()
        
    def compute(self):
        iou_fg = self.fg_inter / self.fg_union if self.fg_union > 0 else 0.0
        iou_bg = self.bg_inter / self.bg_union if self.bg_union > 0 else 0.0
        fb_iou = (iou_fg + iou_bg) / 2.0
        return fb_iou, iou_fg, iou_bg

# Visualization
def save_visualization(collage_img, pred_map, placements, save_path, iou, pos_prompts=None, neg_prompts=None):
    if isinstance(pred_map, torch.Tensor): 
        pred_map = pred_map.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(collage_img)
    
    ref_info = placements['ref']
    rx, ry = ref_info['offset']
    rw, rh = ref_info['curr_size']
    cw, ch = placements['canvas_size']
    
    vis_map = pred_map.copy()
    vis_map[ry:ry+rh, rx:rx+rw] = 0
    
    if vis_map.max() > 0:
        masked = np.ma.masked_where(vis_map == 0, vis_map)
        plt.imshow(masked, cmap='spring', alpha=0.5, interpolation='none')

    ax = plt.gca()
    
    if pos_prompts is not None:
        for idx, box in enumerate(pos_prompts):
            cx, cy, w, h = box
            abs_w, abs_h = w * cw, h * ch
            abs_x0, abs_y0 = (cx * cw) - (abs_w / 2), (cy * ch) - (abs_h / 2)
            
            rect = Rectangle((abs_x0, abs_y0), abs_w, abs_h, linewidth=2, edgecolor='#00FF00', facecolor='none', linestyle='-')
            ax.add_patch(rect)
            if idx == 0:
                ax.text(abs_x0, abs_y0 - 5, "POS BOX", color='#00FF00', fontsize=10, fontweight='bold')
                
    if neg_prompts is not None:
        for idx, box in enumerate(neg_prompts):
            cx, cy, w, h = box
            abs_w, abs_h = w * cw, h * ch
            abs_x0, abs_y0 = (cx * cw) - (abs_w / 2), (cy * ch) - (abs_h / 2)
            
            rect = Rectangle((abs_x0, abs_y0), abs_w, abs_h, linewidth=2, edgecolor='#FF0000', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            if idx == 0:
                ax.text(abs_x0, abs_y0 - 5, "NEG BOX", color='#FF0000', fontsize=10, fontweight='bold')
    
    tgt = placements['tgt']
    tox, toy = tgt['offset']
    ax.text(tox+10, toy+30, f"Local IoU: {iou:.2%}", color='white', fontweight='bold', bbox=dict(facecolor='blue', alpha=0.5))
    
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

# Evaluator
class PairwiseSam3Evaluator:
    def __init__(self, processor, input_height=1008, split_ratio=0.6, max_neg=1):
        self.processor = processor
        self.canvas_size = input_height 
        self.split_ratio = split_ratio
        self.max_neg = max_neg
        self.device = processor.device
        self.metric = ClassWiseIOUMetric()
        self.fb_metric = FBIouMetric() 

    def create_input(self, ref_data, target_data):
        canvas = Image.new('RGB', (self.canvas_size, self.canvas_size), (0, 0, 0))

        s_height = int(self.canvas_size * self.split_ratio)
        t_height = self.canvas_size - s_height

        ref_resized = ref_data['image'].resize((self.canvas_size, s_height), Image.BILINEAR)
        tgt_resized = target_data['image'].resize((self.canvas_size, t_height), Image.BILINEAR)

        canvas.paste(tgt_resized, (0, 0))
        canvas.paste(ref_resized, (0, t_height))
        
        placements = {
            'ref': {'offset': (0, t_height), 'orig_size': ref_data['image'].size, 'curr_size': (self.canvas_size, s_height), 'orig_box': ref_data['box']},
            'tgt': {'offset': (0, 0), 'orig_size': target_data['image'].size, 'curr_size': (self.canvas_size, t_height), 'orig_mask': target_data['mask']},
            'canvas_size': (self.canvas_size, self.canvas_size)
        }
        return canvas, placements

    def run_episode(self, ref_data, target_datas, cat_id, vis_prefix=None, cls_name="visual"):
        if isinstance(ref_data, list):
            ref_data = ref_data[0]

        ref_image = ref_data['image']
        ref_w, ref_h = ref_image.size
        
        bx, by, bw, bh = ref_data['box']
        ref_norm_box = [(bx + bw/2)/ref_w, (by + bh/2)/ref_h, bw/ref_w, bh/ref_h]

        self.processor.set_confidence_threshold(0.0)
        state_s1 = self.processor.set_image(ref_image)
        state_s1 = self.processor.add_geometric_prompt(box=ref_norm_box, label=True, state=state_s1)

        s1_boxes = state_s1['boxes']   
        s1_scores = state_s1['scores'] 

        pos_mask = s1_scores > 0.5
        neg_mask = (s1_scores <= 0.5) & (s1_scores > 0.1)

        pos_boxes_raw = s1_boxes[pos_mask]
        neg_boxes_raw = s1_boxes[neg_mask]
        neg_scores_raw = s1_scores[neg_mask]

        if pos_boxes_raw.shape[0] > 0 and neg_boxes_raw.shape[0] > 0:
            neg_boxes_ext = neg_boxes_raw.unsqueeze(1)
            pos_boxes_ext = pos_boxes_raw.unsqueeze(0)
            
            inter_x1 = torch.max(neg_boxes_ext[:, :, 0], pos_boxes_ext[:, :, 0])
            inter_y1 = torch.max(neg_boxes_ext[:, :, 1], pos_boxes_ext[:, :, 1])
            inter_x2 = torch.min(neg_boxes_ext[:, :, 2], pos_boxes_ext[:, :, 2])
            inter_y2 = torch.min(neg_boxes_ext[:, :, 3], pos_boxes_ext[:, :, 3])
            
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            inter_area = inter_w * inter_h
            
            neg_w = neg_boxes_raw[:, 2] - neg_boxes_raw[:, 0]
            neg_h = neg_boxes_raw[:, 3] - neg_boxes_raw[:, 1]
            neg_area = neg_w * neg_h
            
            ioa = inter_area / (neg_area.unsqueeze(1) + 1e-6)
            max_ioa, _ = ioa.max(dim=1)
            valid_neg_mask = max_ioa <= 0.05
            
            neg_boxes_raw = neg_boxes_raw[valid_neg_mask]
            neg_scores_raw = neg_scores_raw[valid_neg_mask]

        if neg_boxes_raw.shape[0] > 0:
            neg_boxes_raw = neg_boxes_raw.float().contiguous()
            neg_scores_raw = neg_scores_raw.float().contiguous()

            valid_box_mask = (
                (neg_boxes_raw[:, 2] > neg_boxes_raw[:, 0]) & 
                (neg_boxes_raw[:, 3] > neg_boxes_raw[:, 1]) & 
                (~torch.isnan(neg_scores_raw))
            )
            neg_boxes_raw = neg_boxes_raw[valid_box_mask]
            neg_scores_raw = neg_scores_raw[valid_box_mask]

        if neg_boxes_raw.shape[0] > 0:
            sorted_indices = torch.argsort(neg_scores_raw, descending=True)
            neg_boxes_raw = neg_boxes_raw[sorted_indices]
            neg_scores_raw = neg_scores_raw[sorted_indices]

            keep_idx = ops.nms(neg_boxes_raw, neg_scores_raw, iou_threshold=0.5)
            neg_boxes_raw = neg_boxes_raw[keep_idx]

        if neg_boxes_raw.shape[0] > self.max_neg:
            neg_boxes_raw = neg_boxes_raw[:self.max_neg]

        for i, t_data in enumerate(target_datas):
            canvas, placements = self.create_input(ref_data, t_data)
            cw, ch = placements['canvas_size']
            
            s_height = placements['ref']['curr_size'][1]
            t_height = placements['tgt']['curr_size'][1]
            s_ratio = s_height / float(ch)
            t_ratio = t_height / float(ch)

            def map_to_collage(boxes_xyxy, img_w, img_h):
                mapped = []
                for box in boxes_xyxy:
                    x0, y0, x1, y1 = box.tolist()
                    nx0, ny0 = x0 / img_w, y0 / img_h
                    nx1, ny1 = x1 / img_w, y1 / img_h
                    cx = (nx0 + nx1) / 2
                    cy = ((ny0 + ny1) / 2) * s_ratio + t_ratio
                    w = (nx1 - nx0)
                    h = (ny1 - ny0) * s_ratio
                    mapped.append([cx, cy, w, h])
                return mapped

            pos_prompts = map_to_collage(pos_boxes_raw, ref_w, ref_h)
            neg_prompts = map_to_collage(neg_boxes_raw, ref_w, ref_h)

            if len(pos_prompts) == 0:
                cx, cy, w, h = ref_norm_box
                pos_prompts.append([cx, cy * s_ratio + t_ratio, w, h * s_ratio])

            all_boxes = pos_prompts + neg_prompts
            all_labels = [True] * len(pos_prompts) + [False] * len(neg_prompts)

            self.processor.set_confidence_threshold(0.5)
            state_s2 = self.processor.set_image(canvas)
            self.processor.reset_all_prompts(state_s2)

            state_s2 = add_multiple_geometric_prompts(
                processor=self.processor, 
                boxes=all_boxes, 
                labels=all_labels, 
                state=state_s2
            )
            
            pred_masks = state_s2['masks']
            if pred_masks is not None and pred_masks.shape[0] > 0:
                raw_map = torch.max(pred_masks[:, 0], dim=0)[0].int()
            else:
                raw_map = torch.zeros((1008, 1008), dtype=torch.int32, device=self.device)
                
            if raw_map.shape != (ch, cw):
                raw_map = F.interpolate(raw_map[None, None].float(), size=(ch, cw), mode='nearest')[0, 0].int()
            
            tgt = placements['tgt']
            tx, ty = tgt['offset']
            tw, th = tgt['curr_size']
            crop = raw_map[ty:ty+th, tx:tx+tw].cpu().numpy().astype(np.uint8)
            pred_final = np.array(Image.fromarray(crop).resize(tgt['orig_size'], Image.NEAREST))
            
            self.metric.update(pred_final, tgt['orig_mask'], cat_id)
            self.fb_metric.update(pred_final, tgt['orig_mask'])
            
            inter = np.logical_and(pred_final > 0, tgt['orig_mask'] > 0).sum()
            union = np.logical_or(pred_final > 0, tgt['orig_mask'] > 0).sum()
            local_iou = inter / (union + 1e-6)
            
            if vis_prefix and i == 0:
                save_visualization(canvas, raw_map, placements, vis_prefix, local_iou, pos_prompts, neg_prompts)
                
        fb_iou, _, _ = self.fb_metric.compute()
        return self.metric.compute(), fb_iou

# Main
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='SAM3 Few-Shot Segmentation Evaluation')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for evaluation')
    parser.add_argument('--max_neg', type=int, default=1, help='Maximum number of negative prompts to use (default: 1)')
    return parser.parse_args()

def main():
    args = parse_args()
    current_seed = args.seed
    setup_seed(current_seed)

    DATA_ROOT = "/home/tsai091/fsssam3/data/MSCOCO2014" 
    LISTS_ROOT = "/home/tsai091/fsssam3/data/lists/coco/fss_list"
    OUTPUT_VIS_DIR = f"/home/tsai091/fsssam3/vis_results/coco_neg_{args.max_neg}"
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
    print("=========================================================")
    print(" SAM3 FSS Standard Evaluation (Negative Mining)")
    print(f" Max Negative Prompts: {args.max_neg}")
    print("=========================================================\n")

    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(current_dir, ".."))
        bpe_path = os.path.join(root_dir, "assets/bpe_simple_vocab_16e6.txt.gz")
        
        model = build_sam3_image_model(bpe_path=bpe_path)
        processor = Sam3Processor(model)
        print("SAM3 model loaded successfully!\n")
    except Exception as e:
        print(f"Model load failed: {e}"); return

    final_scores = []
    final_fb_scores = []
    
    for fold in range(4):
        print(f"Processing Fold-{fold}")
        
        try:
            # Pass the current seed
            dataset = COCOFSSDataset(
                lists_root=LISTS_ROOT, 
                data_root=DATA_ROOT, 
                fold=fold, 
                k_shot=1, 
                mode='val',
                seed=current_seed, 
            )
        except Exception as e:
            print(f"Skip fold {fold}: {e}"); continue
            
        evaluator = PairwiseSam3Evaluator(processor, split_ratio=0.6, max_neg=args.max_neg)
        pbar = tqdm(range(len(dataset)), desc=f"Fold-{fold}")
        
        for i in pbar:
            try:
                ref_data, target_datas, cat_id, cls_name = dataset[i]
                
                vis_path = None
                if i < 10: 
                    vis_path = os.path.join(OUTPUT_VIS_DIR, f"seed{current_seed}_f{fold}_ep{i}_{cls_name}.jpg")
                
                current_miou, current_fb_iou = evaluator.run_episode(ref_data, target_datas, cat_id, vis_prefix=vis_path, cls_name=cls_name)
                pbar.set_postfix({"mIoU": f"{current_miou:.2%}", "FB-IoU": f"{current_fb_iou:.2%}"})

            except Exception as e:
                print(f"  Episode {i} failed: {e}")
                continue
        
        fold_score = evaluator.metric.compute()
        fold_fb_score, _, _ = evaluator.fb_metric.compute()

        final_scores.append(fold_score)
        final_fb_scores.append(fold_fb_score)
        print(f"Fold-{fold} mIoU: {fold_score:.4%}, FB-IoU: {fold_fb_score:.4%}\n")

    mean_miou = np.mean(final_scores)
    mean_fb_iou = np.mean(final_fb_scores)
    
    print("=========================================================")
    print(" Evaluation Results")
    print("=========================================================")
    for f in range(4):
        print(f" Fold-{f}    : mIoU {final_scores[f]:.4%} | FB-IoU {final_fb_scores[f]:.4%}")
    print("-" * 50)
    print(f" Mean mIoU    : {mean_miou:.4%}")
    print(f" Mean FB-IoU  : {mean_fb_iou:.4%}")
    print("=========================================================")

if __name__ == "__main__":
    main()