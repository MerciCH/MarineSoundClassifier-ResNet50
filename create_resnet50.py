from torchvision import models
from torch import nn

def create_model(device):
    # 使用带预训练权重的 resnet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # 输出 32 类
    model.fc = nn.Linear(model.fc.in_features, 32)
    # 冻结部分层
    for name, param in model.named_parameters():
        if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    return model.to(device)