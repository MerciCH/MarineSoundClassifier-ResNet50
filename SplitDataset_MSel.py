from MyDataset_MSel import MyDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Subset
from skimage.transform import resize

# 定义目标尺寸（如ResNet要求的224x224）
target_size = (224, 224)

# 初始化完整数据集（不带归一化参数）
full_dataset = MyDataset(
    spectrogram_dir=r"WMMS_Spectrogram",
    mfcc_dir=r"WMMS_MFCC",
    melSpec_dir=r"WMMS_MelSpectrogram",  # 新增梅尔频谱图路径
    target_size=target_size,
    transform=None  # 不需要transform，先计算统计量
)

# 计算每个特征的全局均值和标准差
def compute_channel_stats(feature_paths):
    all_features = []
    for path in feature_paths:
        feature = np.load(path).astype(np.float32)
        # 调整尺寸到统一大小（与target_size一致）
        feature = resize(feature, target_size, anti_aliasing=True)
        all_features.append(feature)
    all_features = np.concatenate([f.reshape(1, -1) for f in all_features], axis=0)
    mean = np.mean(all_features)
    std = np.std(all_features)
    return mean, std

# 获取每个特征的路径列表
spectrogram_paths = full_dataset.spectrogram_list_path
mfcc_paths = full_dataset.mfcc_list_path
melSpec_paths = full_dataset.melSpec_list_path

# 计算每个特征的统计量
spectrogram_mean, spectrogram_std = compute_channel_stats(spectrogram_paths)
mfcc_mean, mfcc_std = compute_channel_stats(mfcc_paths)
melSpec_mean, melSpec_std = compute_channel_stats(melSpec_paths)

# 定义 transforms（包含 Resize 和 数据增强）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
])

# 初始化训练集和验证集
train_dataset = MyDataset(
    spectrogram_dir=r"WMMS_Spectrogram",
    mfcc_dir=r"WMMS_MFCC",
    melSpec_dir=r"WMMS_MelSpectrogram",
    spectrogram_mean=spectrogram_mean,
    spectrogram_std=spectrogram_std,
    mfcc_mean=mfcc_mean,
    mfcc_std=mfcc_std,
    melSpec_mean=melSpec_mean,
    melSpec_std=melSpec_std,
    transform=train_transform  # 应用训练集的 transforms
)

val_dataset = MyDataset(
    spectrogram_dir=r"WMMS_Spectrogram",
    mfcc_dir=r"WMMS_MFCC",
    melSpec_dir=r"WMMS_MelSpectrogram",
    spectrogram_mean=spectrogram_mean,
    spectrogram_std=spectrogram_std,
    mfcc_mean=mfcc_mean,
    mfcc_std=mfcc_std,
    melSpec_mean=melSpec_mean,
    melSpec_std=melSpec_std,
    transform=None  # 验证集仅 Resize，不增强
)

# 数据集划分（与之前相同）
indices = list(range(len(full_dataset)))  # full_dataset 是未归一化的完整数据集
labels = full_dataset.labels
train_indices, val_indices = train_test_split(
    indices, test_size=0.2, stratify=labels, random_state=42
)

train_subset = Subset(train_dataset, train_indices)
val_subset = Subset(val_dataset, val_indices)