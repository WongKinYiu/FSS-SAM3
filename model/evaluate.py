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

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import SAM3
try:
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3 import build_sam3_image_model
    from data.dataset_tool import PascalVOCFSSDataset, PASCAL_ID_TO_NAME, COCOFSSDataset, COCO_ID_TO_NAME
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Metrics
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
    
# Visualizations
def save_visualization_1shot(collage_img, pred_map, placements, save_path, iou):
    if isinstance(pred_map, torch.Tensor): 
        pred_map = pred_map.cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.imshow(collage_img)
    
    ref_info = placements['ref']
    rx, ry = ref_info['offset']
    rw, rh = ref_info['curr_size']
    
    vis_map = pred_map.copy()
    vis_map[ry:ry+rh, rx:rx+rw] = 0
    
    if vis_map.max() > 0:
        masked = np.ma.masked_where(vis_map == 0, vis_map)
        plt.imshow(masked, cmap='spring', alpha=0.5, interpolation='none')

    ax = plt.gca()
    bx, by, bw, bh = ref_info['orig_box']
    ox, oy = ref_info['offset']
    sx, sy = ref_info['curr_size'][0]/ref_info['orig_size'][0], ref_info['curr_size'][1]/ref_info['orig_size'][1]
    
    rect = Rectangle((bx*sx + ox, by*sy + oy), bw*sx, bh*sy, linewidth=2, edgecolor='#00FF00', facecolor='none')
    ax.add_patch(rect)
    ax.text(bx*sx + ox, by*sy + oy - 5, "PROMPT BOX", color='#00FF00', fontsize=10, fontweight='bold')
    
    tgt = placements['tgt']
    tox, toy = tgt['offset']
    ax.text(tox+10, toy+30, f"Local IoU: {iou:.2%}", color='white', fontweight='bold', bbox=dict(facecolor='blue', alpha=0.5))
    
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

def save_visualization_5shot(collage_img, pred_map, placements, save_path, iou):
    if isinstance(pred_map, torch.Tensor): 
        pred_map = pred_map.cpu().numpy()
    
    w, h = collage_img.size
    aspect_ratio = w / h
    plt.figure(figsize=(10 * aspect_ratio, 10))
    plt.imshow(collage_img)
    ax = plt.gca()
    
    vis_map = pred_map.copy()
    
    for p in placements:
        if p['type'] == 'ref':
            rx, ry = p['offset']
            rw, rh = p['curr_size']
            vis_map[ry:ry+rh, rx:rx+rw] = 0
            
            bx, by, bw, bh = p['orig_box']
            sx, sy = rw / p['orig_size'][0], rh / p['orig_size'][1]
            rect = Rectangle((bx*sx + rx, by*sy + ry), bw*sx, bh*sy, linewidth=2, edgecolor='#00FF00', facecolor='none')
            ax.add_patch(rect)
            ax.text(bx*sx + rx, by*sy + ry - 5, "PROMPT", color='#00FF00', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
            
    if vis_map.max() > 0:
        masked = np.ma.masked_where(vis_map == 0, vis_map)
        plt.imshow(masked, cmap='spring', alpha=0.5, interpolation='none')

    tgt_p = next(p for p in placements if p['type'] == 'tgt')
    tox, toy = tgt_p['offset']
    ax.text(tox+10, toy+30, f"Local IoU: {iou:.2%}", color='white', fontweight='bold', fontsize=12,
            bbox=dict(facecolor='blue', alpha=0.6))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

# Evaluators
class PairwiseSam3Evaluator:
    def __init__(self, processor, orientation='vertical', split_ratio=0.6, swap_order=False, canvas_size=1008, use_text=True):
        self.processor = processor
        self.device = processor.device
        self.orientation = orientation
        self.split_ratio = split_ratio
        self.swap_order = swap_order
        self.canvas_size = int(canvas_size)
        self.use_text = use_text
        self.metric = ClassWiseIOUMetric()
        self.fb_metric = FBIouMetric()

    def create_input(self, ref_data, target_data):
        canvas = Image.new('RGB', (self.canvas_size, self.canvas_size), (0, 0, 0))
        
        current_orientation = self.orientation

        split_pos = int(self.canvas_size * self.split_ratio)
        rem_pos = self.canvas_size - split_pos
        
        if current_orientation == 'vertical':
            s_rect = (0, 0, self.canvas_size, split_pos)
            t_rect = (0, split_pos, self.canvas_size, rem_pos)
            if self.swap_order: 
                s_rect, t_rect = (0, rem_pos, self.canvas_size, split_pos), (0, 0, self.canvas_size, rem_pos)
        else:
            s_rect = (0, 0, split_pos, self.canvas_size)
            t_rect = (split_pos, 0, rem_pos, self.canvas_size)
            if self.swap_order: 
                s_rect, t_rect = (rem_pos, 0, split_pos, self.canvas_size), (0, 0, rem_pos, self.canvas_size)

        layouts = [
            {'offset': (s_rect[0], s_rect[1]), 'max_dim': (s_rect[2], s_rect[3]), 'data': ref_data, 'type': 'ref'},
            {'offset': (t_rect[0], t_rect[1]), 'max_dim': (t_rect[2], t_rect[3]), 'data': target_data, 'type': 'tgt'}
        ]

        placements = {'canvas_size': (self.canvas_size, self.canvas_size)}
        for lay in layouts:
            img = lay['data']['image']
            target_w, target_h = int(lay['max_dim'][0]), int(lay['max_dim'][1])
            img_resized = img.resize((target_w, target_h), Image.BILINEAR)
            canvas.paste(img_resized, lay['offset'])
            
            placements[lay['type']] = {
                'offset': lay['offset'], 'curr_size': (target_w, target_h),
                'orig_size': img.size, 'orig_box': lay['data'].get('box'), 'orig_mask': lay['data'].get('mask')
            }
        return canvas, placements

    def get_norm_box(self, placements):
        p = placements['ref']
        cw, ch = placements['canvas_size']
        bx, by, bw, bh = p['orig_box']
        ox, oy = p['offset']
        sx, sy = p['curr_size'][0] / p['orig_size'][0], p['curr_size'][1] / p['orig_size'][1]
        px, py = bx * sx + ox, by * sy + oy
        return [(px + bw * sx / 2) / cw, (py + bh * sy / 2) / ch, (bw * sx) / cw, (bh * sy) / ch]

    def run_episode(self, ref_data, target_datas, cat_id, vis_prefix=None, cls_name="visual"):
        if isinstance(ref_data, list):
            ref_data = ref_data[0]

        for i, t_data in enumerate(target_datas):
            canvas, placements = self.create_input(ref_data, t_data)
            norm_box = self.get_norm_box(placements)
            cw, ch = placements['canvas_size']
            
            state = self.processor.set_image(canvas)
            self.processor.reset_all_prompts(state)
            
            if self.use_text:
                state = self.processor.set_text_prompt(cls_name, state)
            
            state = self.processor.add_geometric_prompt(state=state, box=norm_box, label=True)
            
            pred_masks = state['masks']
            if pred_masks.shape[0] > 0:
                raw_map = torch.max(pred_masks[:, 0], dim=0)[0].int()
            else:
                raw_map = torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.int32, device=self.device)
            
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
                save_visualization_1shot(canvas, raw_map, placements, vis_prefix, local_iou)
                
        fb_iou, _, _ = self.fb_metric.compute()
        return self.metric.compute(), fb_iou

class Parameterized5ShotEvaluator:
    def __init__(self, processor, layout_type='l_shape', split_ratio=0.33, canvas_size=1008, use_text=True):
        self.processor = processor
        self.device = processor.device
        self.layout_type = layout_type
        self.split_ratio = split_ratio
        self.canvas_size = int(canvas_size)
        self.use_text = use_text
        self.metric = ClassWiseIOUMetric() 
        self.fb_metric = FBIouMetric()

    def create_canvas(self, ref_datas, target_data):
        canvas = Image.new('RGB', (self.canvas_size, self.canvas_size), (0, 0, 0)) 
        placements = []
        
        s_thick = int(round(self.canvas_size * self.split_ratio))
        t_size = self.canvas_size - s_thick
        
        layouts = []
        if self.layout_type == 'l_shape':
            top_w = self.canvas_size // 3
            left_h = t_size // 2 
            layouts = [
                (0, 0, top_w, s_thick), (top_w, 0, top_w, s_thick), (top_w * 2, 0, self.canvas_size - top_w * 2, s_thick),
                (0, s_thick, s_thick, left_h), (0, s_thick + left_h, s_thick, t_size - left_h),
                (s_thick, s_thick, t_size, t_size)
            ]
        elif self.layout_type == 'vertical_strip':
            sub_h = self.canvas_size // 5
            for i in range(5):
                h = sub_h if i < 4 else self.canvas_size - sub_h * 4
                layouts.append((0, i * sub_h, s_thick, h))
            layouts.append((s_thick, 0, t_size, self.canvas_size))
        elif self.layout_type == 'horizontal_strip':
            sub_w = self.canvas_size // 5
            for i in range(5):
                w = sub_w if i < 4 else self.canvas_size - sub_w * 4
                layouts.append((i * sub_w, 0, w, s_thick))
            layouts.append((0, s_thick, self.canvas_size, t_size))

        all_data = ref_datas + [target_data]
        
        for idx, item in enumerate(all_data):
            is_target = (idx == 5)
            img = item['image'].copy()
            orig_mask = item.get('mask')

            ox, oy, cw, ch = layouts[idx]
            img_w, img_h = img.size
            cw, ch = int(cw), int(ch)
            img_resized = img.resize((cw, ch), Image.BILINEAR)
            canvas.paste(img_resized, (ox, oy))
            
            orig_box = item.get('box')
            scaled_box = None
            if orig_box is not None:
                bx, by, bw, bh = orig_box
                sx, sy = cw / float(img_w), ch / float(img_h)
                scaled_box = [ox + bx*sx, oy + by*sy, bw*sx, bh*sy]
                
            placements.append({
                'type': 'tgt' if is_target else 'ref',
                'offset': (ox, oy), 'orig_size': (img_w, img_h), 'curr_size': (cw, ch),    
                'scaled_box': scaled_box, 'orig_box': orig_box, 'orig_mask': orig_mask
            })
            
        return canvas, placements, (self.canvas_size, self.canvas_size)

    def run_episode(self, ref_datas, target_datas, cat_id, vis_prefix=None, cls_name="visual"):
        for i, t_data in enumerate(target_datas):
            canvas, placements, canvas_size = self.create_canvas(ref_datas, t_data)
            cw, ch = canvas_size
            
            state = self.processor.set_image(canvas)
            self.processor.reset_all_prompts(state)

            if self.use_text:
                state = self.processor.set_text_prompt(cls_name, state)
            
            for p in placements:
                if p['type'] == 'ref':
                    sbx, sby, sbw, sbh = p['scaled_box'] 
                    norm_box = [(sbx + sbw/2)/cw, (sby + sbh/2)/ch, sbw/cw, sbh/ch]
                    state = self.processor.add_geometric_prompt(state=state, box=norm_box, label=True)
            
            pred_masks = state['masks']
            if pred_masks.shape[0] > 0:
                raw_map = torch.max(pred_masks[:, 0], dim=0)[0].int()
            else:
                raw_map = torch.zeros((self.canvas_size, self.canvas_size), dtype=torch.int32, device=self.device)
            
            tgt_p = next(p for p in placements if p['type'] == 'tgt')
            tx, ty = tgt_p['offset']
            tw, th = tgt_p['curr_size']
            crop = raw_map[ty:ty+th, tx:tx+tw].cpu().numpy().astype(np.uint8)
            
            pred_final = np.array(Image.fromarray(crop).resize(tgt_p['orig_size'], Image.NEAREST))
            tgt_orig_mask = tgt_p['orig_mask']
            
            self.metric.update(pred_final, tgt_orig_mask, cat_id)
            self.fb_metric.update(pred_final, tgt_orig_mask)
            
            inter = np.logical_and(pred_final > 0, tgt_orig_mask > 0).sum()
            union = np.logical_or(pred_final > 0, tgt_orig_mask > 0).sum()
            local_iou = inter / (union + 1e-6)
            
            if vis_prefix and i == 0:
                save_visualization_5shot(canvas, raw_map, placements, vis_prefix, local_iou)
                
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
    parser = argparse.ArgumentParser(description='SAM3 FSS Evaluation (Unified)')
    
    # Global arguments
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'], help='Dataset to evaluate on (default: coco)')
    parser.add_argument('--shot', type=int, default=1, choices=[1, 5], help='Number of shots (1 or 5) (default: 1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: 0 for coco, 123 for pascal)') 
    parser.add_argument('--disable_text', action='store_true', help='Disable text prompt (default is Enabled)')
    parser.add_argument('--ratio', type=float, default=None, help='Support area ratio (default: 0.6 for 1-shot, 0.3 for 5-shot)')
    
    # 1-shot specific arguments
    parser.add_argument('--orient', type=str, default='vertical', choices=['vertical', 'horizontal'], help='(1-shot) Orientation')
    parser.add_argument('--swap', action='store_true', help='(1-shot) Force swap (default auto-enabled for coco 1-shot)')
    parser.add_argument('--no_swap', action='store_true', help='(1-shot) Force disable swap')
    
    # 5-shot specific arguments
    parser.add_argument('--layout', type=str, default='l_shape', choices=['l_shape', 'vertical_strip', 'horizontal_strip'], help='(5-shot) Layout type')
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.seed is not None:
        current_seed = args.seed
    else:
        current_seed = 0 if args.dataset == 'coco' else 123
        
    setup_seed(current_seed)
    use_text_prompt = not args.disable_text
    
    current_ratio = args.ratio if args.ratio is not None else (0.6 if args.shot == 1 else 0.3)

    if args.swap:
        current_swap = True
    elif args.no_swap:
        current_swap = False
    else:
        current_swap = (args.dataset == 'coco')

    # Set Paths
    if args.dataset == 'pascal':
        DATA_ROOT = "../data/VOCdevkit/VOC2012"
        LISTS_ROOT = None
        vis_folder_name = f"pascal_{args.shot}shot"
        title_prefix = f"SAM3 PASCAL-5i {args.shot}-Shot Evaluation"
    else:
        DATA_ROOT = "../data/MSCOCO2014" 
        LISTS_ROOT = "../data/lists/coco/fss_list"
        vis_folder_name = f"coco_{args.shot}shot"
        title_prefix = f"SAM3 COCO-20i {args.shot}-Shot Evaluation"
        
    OUTPUT_VIS_DIR = f"../vis_results/{vis_folder_name}"
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)
    
    print("=========================================================")
    print(f" {title_prefix}")
    print("=========================================================\n")

    try:
        bpe_path = os.path.join(root_dir, "assets/bpe_simple_vocab_16e6.txt.gz")
        model = build_sam3_image_model(bpe_path=bpe_path)
        processor = Sam3Processor(model)
        print("SAM3 model loaded successfully!\n")
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    final_scores = []
    final_fb_scores = []
    
    for fold in range(4):
        print(f"Processing Fold-{fold}")
        
        try:
            if args.dataset == 'pascal':
                dataset = PascalVOCFSSDataset(data_root=DATA_ROOT, fold=fold, k_shot=args.shot, mode='val', seed=current_seed)
            else:
                dataset = COCOFSSDataset(lists_root=LISTS_ROOT, data_root=DATA_ROOT, fold=fold, k_shot=args.shot, mode='val', seed=current_seed)
        except Exception as e:
            print(f"Skip fold {fold}: {e}"); continue
            
        # Init Evaluator
        if args.shot == 1:
            evaluator = PairwiseSam3Evaluator(
                processor, orientation=args.orient, split_ratio=current_ratio, 
                swap_order=current_swap, canvas_size=1008, use_text=use_text_prompt
            )
        else:
            evaluator = Parameterized5ShotEvaluator(
                processor, layout_type=args.layout, split_ratio=current_ratio, canvas_size=1008, use_text=use_text_prompt
            )
            
        pbar = tqdm(range(len(dataset)), desc=f"Fold-{fold}")
        
        for i in pbar:
            try:
                if args.dataset == 'pascal':
                    data_out = dataset[i]
                    ref_data, target_datas, cat_id, cls_name = data_out[0], data_out[1], data_out[2], data_out[3]
                else:
                    ref_data, target_datas, cat_id, cls_name = dataset[i]
                
                vis_path = None
                if i < 5: 
                    if args.shot == 1:
                        vis_path = os.path.join(OUTPUT_VIS_DIR, f"f{fold}_ep{i}_{cls_name}.jpg")
                    else:
                        vis_path = os.path.join(OUTPUT_VIS_DIR, f"s{current_seed}_f{fold}_ep{i}_{args.layout}_r{current_ratio}.jpg")
                
                current_miou, current_fb_iou = evaluator.run_episode(ref_data, target_datas, cat_id, vis_prefix=vis_path, cls_name=cls_name)
                pbar.set_postfix({"mIoU": f"{current_miou:.2%}", "FB-IoU": f"{current_fb_iou:.2%}"})
                
            except Exception as e:
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
    
    print(f"RESULT_SUMMARY: mIoU {mean_miou:.4%}, FB-IoU {mean_fb_iou:.4%}")

if __name__ == "__main__":
    main()