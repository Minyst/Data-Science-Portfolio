# convert_model.py
# 목적: HF(PyTorch) MobileViT 세그멘테이션 → ONNX(opset 11, 정적 shape)

import os
import torch                                  # PyTorch (권장: 2.6+)
import onnx                                    # ONNX 로딩/검사
from transformers import MobileViTForSemanticSegmentation

# 0) 경로
project_root = os.path.abspath(os.path.dirname(__file__))
model_dir = project_root                      # pytorch_model.bin, config.json, preprocessor_config.json 위치
output_dir = os.path.join(project_root, "assets", "model")
onnx_path = os.path.join(output_dir, "model.onnx")

os.makedirs(output_dir, exist_ok=True)

# 1) 모델 로드 (safetensors 있으면 우선 사용)
use_safetensors = os.path.exists(os.path.join(model_dir, "model.safetensors"))
model = MobileViTForSemanticSegmentation.from_pretrained(
    model_dir, use_safetensors=use_safetensors
)
model.eval()

# 2) 더미 입력 (정적 1x3x512x512)
dummy_input = torch.randn(1, 3, 512, 512)

# 3) PyTorch → ONNX (opset=11, 정적 shape)
with torch.no_grad():
    torch.onnx.export(
        model, dummy_input, onnx_path,
        opset_version=11,
        input_names=["input"], output_names=["logits"],
        do_constant_folding=True
    )

# 4) ONNX 검사
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

print(f"[OK] ONNX 변환 완료 → {onnx_path}")
