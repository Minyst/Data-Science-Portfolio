# Recycling Segmentation — FINAL (AUG ON/OFF Comparison Version)

# ========== 0) 기본 임포트/재현성 ==========
import os, cv2, random, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
from tqdm import tqdm
import math

# Albumentations는 시도 후 실패 시 비활성화
try:
    import albumentations as A
    HAS_ALB = True
except ImportError:
    HAS_ALB = False

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
set_seed(42)

# ========== 공통 설정: 경로, 클래스, 디바이스 등 ==========
base_dir = "C:/Users/USER/Desktop/Recycling Segmentation.v47i.png-mask-semantic"
results_dir = os.path.join(base_dir, "results"); os.makedirs(results_dir, exist_ok=True)
train_dir = os.path.join(base_dir, "train")
test_dir  = os.path.join(base_dir, "test")  # 지금은 비워둔 상태 (split은 코드에서 처리)

class_names = ["background", "can", "glass", "paper", "plastic", "vinyl"]
num_classes = len(class_names)
label2id = {n:i for i,n in enumerate(class_names)}
id2label = {i:n for n,i in label2id.items()}
class_colors = [(0,0,0),(0,255,255),(255,255,0),(128,255,0),(255,0,0),(255,0,128)]
IMG_H, IMG_W = 512, 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_enabled = (device.type == "cuda")

print(f"📁 Base Dir: {base_dir}")
print(f"🔧 Device: {device} (AMP={amp_enabled}) | InputSize={IMG_H}x{IMG_W}")

# ========== 데이터셋 클래스 및 유틸 함수 ==========
IGNORE_DIR_NAMES = {"results", "best_model"}
def _is_ignored_dir(path):
    return any((p.lower() in IGNORE_DIR_NAMES) or p.lower().startswith("aug")
               for p in os.path.normpath(path).split(os.sep))

def find_image_mask_pairs_recursive(root_dir):
    img_exts = {".jpg",".jpeg",".png",".webp"}
    mask_tokens = ("_mask","-mask","_seg","_label","_labels")
    imgs, masks = {}, {}
    for cur, _, files in os.walk(root_dir):
        if _is_ignored_dir(cur): continue
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            stem = os.path.splitext(fn)[0]
            path = os.path.join(cur, fn)
            if ext in img_exts and not any(t in stem.lower() for t in mask_tokens):
                imgs[stem] = path
            elif any(stem.lower().endswith(t) for t in mask_tokens) and ext in {".png",".jpg",".jpeg"}:
                base = stem
                for t in mask_tokens:
                    if base.endswith(t): base = base[:-len(t)]; break
                masks[base] = path
    keys = sorted(set(imgs) & set(masks))
    if len(set(imgs)-set(masks))>0:  print(f"⚠️ 마스크 없는 이미지: {len(set(imgs)-set(masks))} in {root_dir}")
    if len(set(masks)-set(imgs))>0:  print(f"⚠️ 이미지 없는 마스크: {len(set(masks)-set(imgs))} in {root_dir}")
    return [{"image": imgs[k], "mask": masks[k], "base_name": k} for k in keys]

class AugmentedSegDataset(Dataset):
    def __init__(self, items, processor, split="train", train_aug_transform=None):
        self.items = items
        self.processor = processor
        self.split = split
        self.train_aug_transform = train_aug_transform
        print(f"📂 {split}: {len(items)} items")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rec = self.items[idx]
        img_bgr = cv2.imread(rec["image"])
        msk = cv2.imread(rec["mask"], cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or msk is None:
            raise ValueError(f"Image/mask load failed: {rec['image']} | {rec['mask']}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if img.shape[:2] != msk.shape[:2]:
            msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        if self.split == "train" and self.train_aug_transform is not None:
            out = self.train_aug_transform(image=img, mask=msk)
            img, msk = out["image"], out["mask"]

        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        msk = np.clip(msk, 0, num_classes-1).astype(np.uint8)

        proc = self.processor(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)
        return {
            "pixel_values": proc["pixel_values"].squeeze(0).float(),
            "labels": torch.from_numpy(msk).long(),
            "base_name": rec["base_name"],
        }


class ModelEMA:
    def __init__(self, model, decay=0.997, device=None):
        self.ema = deepcopy(model).to(device if device else next(model.parameters()).device)
        self.decay = decay
        for p in self.ema.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = dict(model.named_parameters())
        for n, p_ema in self.ema.named_parameters():
            p_src = msd[n]
            if p_ema.dtype.is_floating_point: p_ema.data.mul_(d).add_(p_src.data, alpha=1.0-d)
            else: p_ema.data.copy_(p_src.data)
        src_bufs = dict(model.named_buffers())
        for n, b_ema in self.ema.named_buffers():
            b_src = src_bufs[n]
            if b_ema.dtype.is_floating_point: b_ema.copy_(b_src)

# ========== 훈련/평가 전체 과정을 담은 메인 함수 ==========
def run_training_session(use_augmentation: bool):
    """
    하나의 완전한 훈련 및 평가 세션을 실행합니다.
    Args:
        use_augmentation (bool): 훈련 시 데이터 증강을 사용할지 여부.
    Returns:
        dict: 테스트 점수가 담긴 history 딕셔너리.
    """
    set_seed(42)
    run_suffix = "aug_on" if use_augmentation else "aug_off"
    print(f"\n{'='*25}\n Running Session: {run_suffix.upper()} \n{'='*25}")

    # 1. 시나리오별 결과 경로 설정
    best_model_dir   = os.path.join(results_dir, f"best_model_{run_suffix}"); os.makedirs(best_model_dir, exist_ok=True)
    final_model_dir  = os.path.join(results_dir, f"final_model_{run_suffix}"); os.makedirs(final_model_dir, exist_ok=True)
    test_vis_agg_dir = os.path.join(results_dir, f"test_visuals_agg_{run_suffix}"); os.makedirs(test_vis_agg_dir, exist_ok=True)
    test_scores_path = os.path.join(results_dir, f"test_scores_{run_suffix}.txt")
    perclass_path    = os.path.join(results_dir, f"test_per_class_{run_suffix}.txt")

    # 2. 데이터 증강 설정 (실무 표준)
    train_aug_transform = None
    if use_augmentation and HAS_ALB:
        print("✨ Albumentations 활성화 (Train 전용: 균형 잡힌 3가지)")
        train_aug_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=0.4),
            A.ColorJitter(brightness=0.18, contrast=0.08, saturation=0.08, hue=0.03, p=0.3),
        ], p=1.0)
    else:
        print(f"❌ 증강 비활성화 (요청: {use_augmentation}, 설치됨: {HAS_ALB})")

    # 3. 데이터 로딩 + 150/50 split (train_dir 기준)
    #    - train_dir에 200쌍이 들어있다고 가정
    all_pairs = find_image_mask_pairs_recursive(train_dir)
    print(f"📂 train_dir 전체 데이터: {len(all_pairs)}개")

    set_seed(42)                    # 항상 같은 split을 위해 고정
    random.shuffle(all_pairs)       # 섞은 뒤 상위 50개를 test로 사용
    test_size = 50
    test_pairs  = all_pairs[:test_size]
    train_pairs = all_pairs[test_size:]

    print(f"📂 Split 결과 → Train: {len(train_pairs)}개 | Test: {len(test_pairs)}개 (목표: 150 / 50)")

    all_pairs_for_lookup = train_pairs + test_pairs
    base_lookup = {r["base_name"]: r for r in all_pairs_for_lookup}

    train_items = train_pairs.copy()
    # TTA 제거 - aug_on, aug_off 둘 다 원본만 사용
    test_items = [{**rec, "rotation": 0, "flip": False} for rec in test_pairs]
    print(f"📂 Test: {len(test_items)}개 (TTA 없음, 원본만 사용)")

    base_ckpt = "apple/deeplabv3-mobilevit-x-small"
    processor = AutoImageProcessor.from_pretrained(base_ckpt, use_fast=True)

    # 샘플러 없이, multiplier 기반으로 에폭당 샘플 수 설정
    BATCH_SIZE = 16
    # aug_on일 때만 5배 multiplier 적용, aug_off일 때는 1배
    multiplier = 5 if use_augmentation else 1
    # 🔧 여기서 정확히 len(train_items) * multiplier 로 맞춤 (예: 150 * 5 = 750)
    TARGET_N = len(train_items) * multiplier
    train_items_ext = train_items
    if TARGET_N and TARGET_N > len(train_items_ext):
        repeat = math.ceil(TARGET_N / len(train_items_ext))
        train_items_ext = (train_items_ext * repeat)[:TARGET_N]
    print(f"📂 train unique: {len(train_items)} | effective/epoch: {len(train_items_ext)} (TARGET_N={TARGET_N}, multiplier={multiplier})")

    train_ds = AugmentedSegDataset(train_items_ext, processor, "train", train_aug_transform)
    test_ds  = AugmentedSegDataset(test_items, processor, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=(device.type=="cuda"))
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=(device.type=="cuda"))

    # 4. 모델, 손실함수, 옵티마이저 등 (세션마다 새로 초기화)
    model = AutoModelForSemanticSegmentation.from_pretrained(
        base_ckpt, num_labels=num_classes, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    ).to(device)

    # (이하 유틸 함수들)
    def calc_class_weights(train_base_set, all_pairs):
        weights_by_class = {
            "background": 0.80,
            "can": 1.50,
            "glass": 1.05,
            "paper": 1.65,
            "plastic": 1.45,
            "vinyl": 1.00,
        }
        weights = [float(weights_by_class.get(class_names[c], 1.0)) for c in range(num_classes)]
        return torch.tensor(weights, dtype=torch.float32).to(device)

    class CombinedLoss(torch.nn.Module):
        def __init__(self, w=None, dice_w=0.6, ce_w=0.4, eps=1e-7):
            super().__init__(); self.w=w; self.dice_w=dice_w; self.ce_w=ce_w; self.eps=eps
        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, weight=self.w)
            probs = F.softmax(logits, dim=1); dices=[]
            for c in range(1, logits.size(1)):
                t=(targets==c).float(); p=probs[:,c]
                inter=(p*t).sum(dim=(1,2)); union=p.sum(dim=(1,2))+t.sum(dim=(1,2))
                dices.append(1-((2*inter+self.eps)/(union+self.eps)))
            dice = torch.stack(dices, dim=1).mean() if dices else logits.new_tensor(0.0)
            return self.ce_w*ce + self.dice_w*dice

    def mean_dice_or_iou(preds, targets, metric='dice'):
        scores=[]
        for p,t in zip(preds, targets):
            p=p.flatten(); t=t.flatten(); valid=(t!=0); p=p[valid]; t=t[valid]
            if len(t)==0: continue
            cls_scores=[]
            for c in range(1, num_classes):
                pc=(p==c); tc=(t==c); inter=(pc&tc).sum()
                if metric == 'dice':
                    denom=pc.sum()+tc.sum()
                    if denom > 0: cls_scores.append((2*inter)/denom)
                else: # iou
                    union=(pc|tc).sum()
                    if union > 0: cls_scores.append(inter/union)
            if cls_scores: scores.append(float(np.mean(cls_scores)))
        return float(np.mean(scores)) if scores else 0.0

    def per_class_metrics(preds, targets):
        per_cls_dice, per_cls_iou = {c:[] for c in range(1,num_classes)}, {c:[] for c in range(1,num_classes)}
        for p,t in zip(preds,targets):
            for c in range(1, num_classes):
                pc=(p==c); tc=(t==c); inter=(pc&tc).sum()
                denom=pc.sum()+tc.sum(); union=(pc|tc).sum()
                if denom>0: per_cls_dice[c].append((2*inter)/denom)
                if union>0: per_cls_iou[c].append(inter/union)
        dice_avg = {c:(float(np.mean(v)) if v else 0.0) for c,v in per_cls_dice.items()}
        iou_avg = {c:(float(np.mean(v)) if v else 0.0) for c,v in per_cls_iou.items()}
        return dice_avg, iou_avg

    def mask_to_color(mask):
        h,w=mask.shape; color=np.zeros((h,w,3),dtype=np.uint8)
        for cid in range(num_classes):
            if cid < len(class_colors): color[mask==cid] = class_colors[cid]
        return color

    def overlay_image(rgb, mask, alpha=0.4):
        color=mask_to_color(mask).astype(np.float32); out=rgb.astype(np.float32).copy()
        fg=(mask>0); out[fg]=out[fg]*(1-alpha)+color[fg]*alpha
        return np.clip(out,0,255).astype(np.uint8)


    train_base_set = set(r["base_name"] for r in train_pairs)
    class_weights = calc_class_weights(train_base_set, all_pairs_for_lookup)
    print({class_names[i]: float(class_weights[i]) for i in range(num_classes)})
    criterion = CombinedLoss(w=class_weights, dice_w=0.6, ce_w=0.4)

    max_epochs, warmup_epochs = 300, 5 # 실무 표준
    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=5e-5)  # 원래 학습률로 복원
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
        CosineAnnealingLR(optimizer, T_max=max_epochs-warmup_epochs, eta_min=1e-6)
    ], milestones=[warmup_epochs])
    scaler = GradScaler('cuda', enabled=amp_enabled)
    ema = ModelEMA(model, decay=0.997, device=device)

    # 5. 훈련 루프
    history = defaultdict(list)
    print(f"🚀 훈련 시작! ({run_suffix.upper()})"); print("-" * 80)

    best_train_dice = -1.0; best_epoch = 0
    patience = 40; patience_counter = 0  # EarlyStopping 설정

    for epoch in range(1, max_epochs + 1):
        model.train(); train_losses, train_preds, train_tgts = [], [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d} - Train"):
            imgs, masks = batch["pixel_values"].to(device), batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda', enabled=amp_enabled):
                logits = model(pixel_values=imgs).logits
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            ema.update(model)
            train_losses.append(loss.item())
            with torch.no_grad():
                train_preds.extend(list(logits.argmax(1).cpu().numpy()))
                train_tgts.extend(list(masks.cpu().numpy()))

        scheduler.step()
        train_dice = mean_dice_or_iou(train_preds, train_tgts, 'dice')
        train_miou = mean_dice_or_iou(train_preds, train_tgts, 'iou')

        history["epochs"].append(epoch)
        history["train_loss"].append(np.mean(train_losses))
        history["train_dice"].append(train_dice); history["train_iou"].append(train_miou)

        is_best = train_dice > best_train_dice
        print(f"Epoch {epoch:03d} | TrainLoss {np.mean(train_losses):.4f} | TrainDice {train_dice:.4f} | TrainmIoU {train_miou:.4f} {'🏆' if is_best else ''}")

        # 베스트 모델 저장 (Train Dice 기준, EMA 사용)
        if is_best:
            prev_best = best_train_dice
            best_train_dice, best_epoch = train_dice, epoch
            patience_counter = 0  # 개선되었으므로 카운터 리셋
            print(f"   🎯 NEW BEST TrainDice: {prev_best:.4f} → {best_train_dice:.4f} (epoch {best_epoch})")
            torch.save(ema.ema.state_dict(), os.path.join(best_model_dir, "pytorch_model.bin"))
            ema.ema.config.save_pretrained(best_model_dir); processor.save_pretrained(best_model_dir)
        else:
            patience_counter += 1
            print(f"   ⏳ No improvement for {patience_counter}/{patience} epochs")

        # EarlyStopping 체크
        if patience_counter >= patience:
            print(f"🛑 EarlyStopping triggered! No improvement for {patience} epochs.")
            print(f"   Best TrainDice: {best_train_dice:.4f} at epoch {best_epoch}")
            break

    # 6. 마지막 에폭 모델 저장
    torch.save(ema.ema.state_dict(), os.path.join(final_model_dir, "pytorch_model.bin"))
    ema.ema.config.save_pretrained(final_model_dir); processor.save_pretrained(final_model_dir)

    # 7. 최종 테스트 및 시각화 (세션별로 실행)
    print(f"\n🧪 최종 테스트 ({run_suffix.upper()})")
    eval_model = ema.ema.eval().to(device)
    prob_accum, cnt_accum = defaultdict(lambda: None), defaultdict(int)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Final Test ({run_suffix.upper()})"):
            imgs, bases = batch["pixel_values"].to(device), batch["base_name"]
            with autocast('cuda', enabled=amp_enabled):
                logits = eval_model(pixel_values=imgs).logits
                logits = F.interpolate(logits, size=(IMG_H, IMG_W), mode="bilinear", align_corners=False)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            for i in range(probs.shape[0]):
                bn_i = bases[i]
                pr = probs[i]
                prob_accum[bn_i] = pr.copy() if prob_accum[bn_i] is None else prob_accum[bn_i] + pr
                cnt_accum[bn_i] += 1

    test_preds_base, test_tgts_base = [], []
    for bn, acc in prob_accum.items():
        pred = (acc/cnt_accum[bn]).argmax(axis=0).astype(np.uint8)
        test_preds_base.append(pred)
        rec = base_lookup.get(bn)
        gt = cv2.imread(rec["mask"], cv2.IMREAD_GRAYSCALE) if rec else np.zeros_like(pred)
        gt = cv2.resize(gt, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
        test_tgts_base.append(np.clip(gt, 0, num_classes-1).astype(np.uint8))

    # 7-1. 먼저 테스트 점수 계산/출력/저장
    test_dice_last = mean_dice_or_iou(test_preds_base, test_tgts_base, 'dice')
    test_miou_last = mean_dice_or_iou(test_preds_base, test_tgts_base, 'iou')
    dice_pc, iou_pc = per_class_metrics(test_preds_base, test_tgts_base)

    print(f"✅ FINAL ({run_suffix.upper()}) | TestDice {test_dice_last:.4f} | TestmIoU {test_miou_last:.4f}")
    with open(test_scores_path, "w", encoding="utf-8") as f:
        f.write(f"TestDice(EMA) {test_dice_last:.4f}\nTestmIoU(EMA) {test_miou_last:.4f}\n")
    with open(perclass_path, "w", encoding="utf-8") as f:
        for c in range(1, num_classes):
            f.write(f"{class_names[c]:10s}  Dice {dice_pc[c]:.4f}  IoU {iou_pc[c]:.4f}\n")

    # 7-2. 그 다음 시각화 저장 (원본/GT/Pred/Overlay 4분할)
    def save_quadrant_visual(bn, rgb, gt_mask, pred_mask, out_dir):
        try:
            h, w = rgb.shape[:2]
            gt_color = mask_to_color(gt_mask)
            pred_color = mask_to_color(pred_mask)
            overlay = overlay_image(rgb, pred_mask)

            pad = 4
            tile_h, tile_w = h + pad, w + pad
            canvas = np.zeros((tile_h*2, tile_w*2, 3), dtype=np.uint8)

            canvas[0:h, 0:w] = rgb
            canvas[0:h, tile_w:tile_w+w] = gt_color
            canvas[tile_h:tile_h+h, 0:w] = pred_color
            canvas[tile_h:tile_h+h, tile_w:tile_w+w] = overlay

            out_path = os.path.join(out_dir, f"{bn}_grid.png")
            cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"🔥 시각화 저장 실패: {bn}, {e}")

    for bn, acc in prob_accum.items():
        rec = base_lookup.get(bn)
        pred = (acc/cnt_accum[bn]).argmax(axis=0).astype(np.uint8)
        if rec:
            rgb = cv2.cvtColor(cv2.imread(rec["image"]), cv2.COLOR_BGR2RGB)
            gt = cv2.imread(rec["mask"], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            gt = np.clip(gt, 0, num_classes-1).astype(np.uint8)
            save_quadrant_visual(bn, rgb, gt, pred, test_vis_agg_dir)

    print(f"💾 결과 저장 완료: {test_scores_path}")

    # 최종 테스트 점수만 history에 기록
    history["test_dice_final"] = test_dice_last
    history["test_miou_final"] = test_miou_last
    return history


# ========== 최종 비교 그래프 생성 함수 ==========
def plot_comparison_curves(history_aug, history_no_aug, save_path):
    """두 세션의 에폭별 학습 성능을 비교하는 단일 선 그래프를 생성합니다."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Train Dice Score Curve - 하나의 그래프에 두 선
    ax.plot(history_no_aug.get("epochs", []), history_no_aug.get("train_dice", []),
            color='blue', linewidth=2, label='Aug_Off')
    ax.plot(history_aug.get("epochs", []), history_aug.get("train_dice", []),
            color='red', linewidth=2, label='Aug_On')

    ax.set_title('Train Dice Score Curve', fontsize=9)
    ax.set_xlabel('Epoch', fontsize=8)
    ax.set_ylabel('Dice Score', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n📊 최종 비교 그래프 저장 완료: {save_path}")


# ========== 요약 출력 함수 ==========
def print_summary(results_dir, comparison_plot_path):
    print("\n" + "=" * 50)
    print("✅ 모든 작업이 성공적으로 완료되었습니다.")
    print("📁 결과 위치:")
    print(f" - Aug ON 모델: {os.path.join(results_dir, 'best_model_aug_on')}")
    print(f" - Aug OFF 모델: {os.path.join(results_dir, 'best_model_aug_off')}")
    print(f" - 비교 그래프(Test 기준): {comparison_plot_path}")
    print(f" - 각 시나리오별 점수/시각화 자료는 'results' 폴더 내에 저장되었습니다.")
    print("=" * 50)

# ========== 메인 실행 블록 ==========
if __name__ == "__main__":
    # 세션 1: 데이터 증강 ON (Train 150 → effective 750, Test 50)
    history_aug = run_training_session(use_augmentation=True)

    # 세션 2: 데이터 증강 OFF (Train 150 → effective 150, Test 50)
    history_no_aug = run_training_session(use_augmentation=False)

    # 두 세션의 결과로 최종 비교 그래프 생성
    comparison_plot_path = os.path.join(results_dir, "comparison_curves.png")
    plot_comparison_curves(history_aug, history_no_aug, comparison_plot_path)

    # 최종 요약 출력 (한 번만)
    print_summary(results_dir, comparison_plot_path)
