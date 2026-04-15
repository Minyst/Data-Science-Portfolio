# train_yolo11_seg.py
# 목적: YOLO11n-seg 모델을 재활용 분류 데이터셋으로 fine-tuning
# 단계: 1) 데이터 변환 (PNG 마스크 -> YOLO seg 형식)  2) 학습  3) 평가

import os
import sys
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import yaml

# Windows 콘솔 인코딩 설정
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = "C:/Users/USER/Desktop/Recycling Segmentation.v47i.png-mask-semantic/train"
YOLO_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset")

# 클래스 정보
CLASS_NAMES = ["can", "glass", "paper", "plastic", "vinyl"]  # background 제외
NUM_CLASSES = len(CLASS_NAMES)


def find_image_mask_pairs(root_dir):
    """이미지-마스크 쌍 찾기"""
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    mask_tokens = ("_mask", "-mask", "_seg", "_label", "_labels")
    imgs, masks = {}, {}

    for fn in os.listdir(root_dir):
        path = os.path.join(root_dir, fn)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(fn)[1].lower()
        stem = os.path.splitext(fn)[0]

        if ext in img_exts and not any(t in stem.lower() for t in mask_tokens):
            imgs[stem] = path
        elif any(stem.lower().endswith(t) for t in mask_tokens) and ext in {".png", ".jpg", ".jpeg"}:
            base = stem
            for t in mask_tokens:
                if base.endswith(t):
                    base = base[:-len(t)]
                    break
            masks[base] = path

    keys = sorted(set(imgs) & set(masks))
    return [{"image": imgs[k], "mask": masks[k], "base_name": k} for k in keys]


def mask_to_yolo_segments(mask, class_id):
    """
    바이너리 마스크를 YOLO 세그멘테이션 형식 (정규화된 폴리곤 좌표)으로 변환
    Returns: list of polygon strings
    """
    segments = []

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape

    for contour in contours:
        # 너무 작은 컨투어 무시 (최소 3점 필요)
        if len(contour) < 3:
            continue

        # 면적이 너무 작으면 무시
        area = cv2.contourArea(contour)
        if area < 100:  # 최소 면적 threshold
            continue

        # 폴리곤 단순화 (점 개수 줄이기)
        epsilon = 0.005 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) < 3:
            continue

        # 정규화된 좌표로 변환
        points = []
        for point in approx:
            x = point[0][0] / w
            y = point[0][1] / h
            points.extend([x, y])

        # YOLO 형식: class_id x1 y1 x2 y2 ...
        segment_str = f"{class_id} " + " ".join([f"{p:.6f}" for p in points])
        segments.append(segment_str)

    return segments


def convert_dataset_to_yolo(pairs, output_dir, split_name="train"):
    """데이터셋을 YOLO 세그멘테이션 형식으로 변환"""

    images_dir = os.path.join(output_dir, "images", split_name)
    labels_dir = os.path.join(output_dir, "labels", split_name)

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    converted_count = 0
    skipped_count = 0

    for item in tqdm(pairs, desc=f"Converting {split_name}"):
        # 이미지 복사
        img_name = os.path.basename(item["image"])
        img_stem = os.path.splitext(img_name)[0]

        # 안전한 파일명으로 변경
        safe_stem = img_stem.replace("-", "_").replace(" ", "_")

        dst_img_path = os.path.join(images_dir, f"{safe_stem}.jpg")

        # 이미지 읽고 저장 (jpg로 통일)
        img = cv2.imread(item["image"])
        if img is None:
            skipped_count += 1
            continue
        cv2.imwrite(dst_img_path, img)

        # 마스크 읽기
        mask = cv2.imread(item["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped_count += 1
            continue

        # 라벨 파일 생성
        label_path = os.path.join(labels_dir, f"{safe_stem}.txt")

        all_segments = []

        # 각 클래스별로 세그먼트 추출 (background=0 제외, 1~5)
        for class_idx in range(1, 6):  # can(1), glass(2), paper(3), plastic(4), vinyl(5)
            class_mask = (mask == class_idx).astype(np.uint8) * 255

            if class_mask.sum() > 0:
                # YOLO 클래스 인덱스는 0부터 시작
                yolo_class_id = class_idx - 1
                segments = mask_to_yolo_segments(class_mask, yolo_class_id)
                all_segments.extend(segments)

        # 라벨 파일 저장
        with open(label_path, "w") as f:
            f.write("\n".join(all_segments))

        converted_count += 1

    print(f"[OK] {split_name}: {converted_count}개 변환 완료, {skipped_count}개 스킵")
    return converted_count


def create_yaml_config(output_dir):
    """YOLO 데이터셋 설정 파일 생성"""
    config = {
        "path": output_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "can",
            1: "glass",
            2: "paper",
            3: "plastic",
            4: "vinyl"
        }
    }

    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"[OK] YAML 설정 파일 생성: {yaml_path}")
    return yaml_path


def train_yolo11_seg(yaml_path, epochs=300, imgsz=512, batch=16):
    """YOLO11n-seg 모델 학습 - MobileViT와 동일 조건"""
    from ultralytics import YOLO

    print("\n" + "="*60)
    print("[INFO] YOLO11n-seg Fine-tuning 시작")
    print("[INFO] MobileViT와 동일 조건: 300 epochs, batch=16, 증강 적용")
    print("="*60)

    # 사전학습된 YOLO11n-seg 모델 로드
    model = YOLO("yolo11n-seg.pt")

    # 학습 실행 - MobileViT와 동일한 조건
    # MobileViT: 300 epochs, batch=16, EarlyStopping patience=40
    # 증강: HorizontalFlip(0.5), Rotate(10도, 0.4), ColorJitter(0.3)
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=40,  # MobileViT와 동일한 EarlyStopping patience
        save=True,
        project=os.path.join(PROJECT_ROOT, "yolo_runs"),
        name="recycling_seg",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        device=0 if __import__("torch").cuda.is_available() else "cpu",
        workers=0,  # Windows 호환
        amp=True,
        verbose=True,
        # 증강 설정 - MobileViT와 유사하게
        flipud=0.0,        # 상하 뒤집기 OFF
        fliplr=0.5,        # 좌우 뒤집기 (HorizontalFlip p=0.5)
        degrees=10.0,      # 회전 각도 (Rotate limit=10)
        hsv_h=0.015,       # 색상 변화 (ColorJitter hue=0.03의 절반)
        hsv_s=0.4,         # 채도 변화 (ColorJitter saturation=0.08 확대)
        hsv_v=0.4,         # 밝기 변화 (ColorJitter brightness=0.18 확대)
        mosaic=0.0,        # 모자이크 OFF (MobileViT에 없음)
        mixup=0.0,         # 믹스업 OFF (MobileViT에 없음)
        copy_paste=0.0,    # 복사붙여넣기 OFF
    )

    print("[OK] 학습 완료!")
    return model, results


def main():
    print("\n" + "="*70)
    print("    YOLO11n-seg Fine-tuning for Recycling Classification")
    print("="*70)

    # 1. 데이터 로드
    print("\n[Step 1] 데이터 로드 및 분할")
    all_pairs = find_image_mask_pairs(SOURCE_DIR)
    print(f"  - 전체 데이터: {len(all_pairs)}개")

    # 데이터 분할 (150 train, 50 val) - MobileViT와 동일한 분할
    import random
    random.seed(42)
    random.shuffle(all_pairs)

    val_pairs = all_pairs[:50]
    train_pairs = all_pairs[50:]

    print(f"  - Train: {len(train_pairs)}개")
    print(f"  - Val: {len(val_pairs)}개")

    # 2. YOLO 형식으로 변환
    print("\n[Step 2] YOLO 세그멘테이션 형식으로 변환")

    # 기존 데이터셋 디렉토리 삭제
    if os.path.exists(YOLO_DATASET_DIR):
        shutil.rmtree(YOLO_DATASET_DIR)
    os.makedirs(YOLO_DATASET_DIR)

    convert_dataset_to_yolo(train_pairs, YOLO_DATASET_DIR, "train")
    convert_dataset_to_yolo(val_pairs, YOLO_DATASET_DIR, "val")

    # 3. YAML 설정 파일 생성
    print("\n[Step 3] 데이터셋 설정 파일 생성")
    yaml_path = create_yaml_config(YOLO_DATASET_DIR)

    # 4. 학습 실행 (MobileViT와 동일 조건)
    print("\n[Step 4] YOLO11n-seg 학습 (MobileViT와 동일 조건)")
    model, results = train_yolo11_seg(
        yaml_path,
        epochs=300,
        imgsz=512,
        batch=16
    )

    # 5. 최종 모델 경로 출력
    best_model_path = os.path.join(PROJECT_ROOT, "yolo_runs", "recycling_seg", "weights", "best.pt")
    print(f"\n[OK] 학습 완료!")
    print(f"  - Best 모델: {best_model_path}")

    return best_model_path


if __name__ == "__main__":
    main()
