from torchvision import models
from torch import nn
def create_vgg16(device, num_classes=32):
    # 使用带预训练权重的 vgg16
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # 冻结 features 部分的所有参数
    for param in model.features.parameters():
        param.requires_grad = False
    # 解冻最后两个卷积块
    for param in model.features[24:].parameters():
        param.requires_grad = True

    # 修改 classifier，让其输出 32 类（我的数据集类别数）
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(1024, num_classes)
    )
    return model.to(device)