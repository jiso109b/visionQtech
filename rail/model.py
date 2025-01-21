import torch
import torch.nn as nn
from torch.nn.utils import prune  # 프루닝
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision import models
import os

original_model_path=r"C:\Users\jump0\Desktop\ai_cv\cv_v2\model\yolov5"

# 커스텀 백본 정의
class CustomBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50 = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])

    def forward(self, x):
        return self.backbone(x)

# YOLOv5 멀티 클래스 모델 정의
class YOLOv5MultiClass(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = CustomBackbone()
        self.multi_class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.multi_class_head(x)
        return x

# 모델 로드 및 초기화
def load_and_prepare_model(original_model_path, num_classes, device):
    # 모델 초기화
    model = YOLOv5MultiClass(num_classes).to(device)
    if not os.path.exists(original_model_path):
        raise FileNotFoundError(f"[ERROR] 모델 파일을 찾을 수 없습니다: {original_model_path}")

    # 모델 로드
    model.load_state_dict(torch.load(original_model_path, map_location=device))
    model.eval()
    print(f"[INFO] 모델 로드 완료: {original_model_path}")

    # 프루닝 적용
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):  # Linear Layer에 대해 프루닝
            prune.l1_unstructured(module, name='weight', amount=0.3)  # 30% 가중치 제거
            prune.remove(module, 'weight')  # 프루닝 후 가중치 재구성
    print("[INFO] 모델 프루닝 적용")

    # TorchScript 변환 및 저장
    tscr_model = torch.jit.script(model)
    pruned_model_path = os.path.join(os.path.dirname(original_model_path), "pruned_scripted_model.pt")
    tscr_model.save(pruned_model_path)
    print(f"[INFO] TorchScript 모델 저장: {pruned_model_path}")

    return model, pruned_model_path

# 전처리 파이프라인 생성
def create_transform():
    transform = Compose([
        Resize((512, 512)),  # 이미지 크기 조정
        ToTensor(),          # 이미지를 Tensor로 변환
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 정규화
    ])
    print("[INFO] 전처리 파이프라인 준비 완료")
    return transform
