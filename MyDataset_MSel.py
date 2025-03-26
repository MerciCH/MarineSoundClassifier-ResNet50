from torch.utils.data import Dataset
import os
import numpy as np
import torch
from skimage.transform import resize


class MyDataset(Dataset):
    def __init__(self, spectrogram_dir, mfcc_dir, melSpec_dir,
                 spectrogram_mean=None, spectrogram_std=None,
                 mfcc_mean=None, mfcc_std=None,
                 melSpec_mean=None, melSpec_std=None,
                 transform=None, target_size=(224, 224)):
        # 初始化路径和其他参数
        self.spectrogram_dir = spectrogram_dir
        self.mfcc_dir = mfcc_dir
        self.melSpec_dir = melSpec_dir

        self.transform = transform  # 数据增强
        self.target_size = target_size  # 目标尺寸

        # 归一化参数（每个特征独立）
        self.spectrogram_mean = spectrogram_mean
        self.spectrogram_std = spectrogram_std
        self.mfcc_mean = mfcc_mean
        self.mfcc_std = mfcc_std
        self.melSpec_mean = melSpec_mean
        self.melSpec_std = melSpec_std

        # 获取所有路径和标签
        self.label_list = sorted(os.listdir(spectrogram_dir))
        self.spectrogram_list_path = []
        self.mfcc_list_path = []
        self.melSpec_list_path = []
        self.labels = []

        for label_idx, label in enumerate(self.label_list):
            spectrogram_path = os.path.join(spectrogram_dir, label)
            mfcc_path = os.path.join(mfcc_dir, label)
            melSpec_path = os.path.join(melSpec_dir, label)

            spectrogram_paths = [os.path.join(spectrogram_path, f)
                                 for f in os.listdir(spectrogram_path) if f.endswith('.npy')]
            mfcc_paths = [os.path.join(mfcc_path, f)
                          for f in os.listdir(mfcc_path) if f.endswith('.npy')]
            melSpec_paths = [os.path.join(melSpec_path, f)
                             for f in os.listdir(melSpec_path) if f.endswith('.npy')]

            self.spectrogram_list_path.extend(spectrogram_paths)
            self.mfcc_list_path.extend(mfcc_paths)
            self.melSpec_list_path.extend(melSpec_paths)
            self.labels.extend([label_idx] * len(spectrogram_paths))

    def __getitem__(self, idx):
        # 加载特征
        spectrogram = np.load(self.spectrogram_list_path[idx]).astype(np.float32)
        mfcc = np.load(self.mfcc_list_path[idx]).astype(np.float32)
        melSpec = np.load(self.melSpec_list_path[idx]).astype(np.float32)

        # 调整尺寸到目标大小（H, W）
        spectrogram = resize(spectrogram, self.target_size, anti_aliasing=True)
        mfcc = resize(mfcc, self.target_size, anti_aliasing=True)
        melSpec = resize(melSpec, self.target_size, anti_aliasing=True)

        # 应用独立归一化
        if self.spectrogram_mean is not None and self.spectrogram_std is not None:
            spectrogram = (spectrogram - self.spectrogram_mean) / self.spectrogram_std
        if self.mfcc_mean is not None and self.mfcc_std is not None:
            mfcc = (mfcc - self.mfcc_mean) / self.mfcc_std
        if self.melSpec_mean is not None and self.melSpec_std is not None:
            melSpec = (melSpec - self.melSpec_mean) / self.melSpec_std

        # 扩展维度并拼接为 [H, W, 3]
        spectrogram = np.expand_dims(spectrogram, axis=-1)
        mfcc = np.expand_dims(mfcc, axis=-1)
        melSpec = np.expand_dims(melSpec, axis=-1)
        fused_feature = np.concatenate([spectrogram, mfcc, melSpec], axis=-1)

        # 转为张量并调整维度为 [C, H, W]（此时形状为 [3, H, W]）
        fused_tensor = torch.from_numpy(fused_feature).permute(2, 0, 1).float()

        # 应用 transforms（如数据增强）
        if self.transform is not None:
            fused_tensor = self.transform(fused_tensor)

        label = self.labels[idx]
        return fused_tensor, label

    def __len__(self):
        return len(self.spectrogram_list_path)