import librosa
import numpy as np
import os
import glob


def save_mfcc(wav_path, save_root, n_mfcc=15):
    # 加载音频，强制指定采样率 sr 为 16KHz，否则后续的 mfcc 提取会报错
    y, sr = librosa.load(wav_path, sr=16000)
    # 提取MFCC，部分音频只有907个样本点，FFT 傅里叶变换窗口不能比实际信号长度还长
    # 因此设置 n_fft=512 n_mels=30 fmax <= sr/2
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, n_mels=30, fmax=7500)

    # 构建新路径（保留原文件夹结构）
    # 音频目录的相对路径（包含文件夹名字和数据名字，但不好含start）
    rel_path = os.path.relpath(wav_path, start="Watkins Marine Mammal Sound (WMMS) Database")
    # 依据文件夹名（类别名）的新目录（仅包含文件夹名字的新目录）
    new_dir = os.path.join(save_root, os.path.dirname(rel_path))
    # 创建目录
    os.makedirs(new_dir, exist_ok=True)

    # 保存MFCC为.npy文件
    save_name = os.path.basename(wav_path).replace(".wav", ".npy")
    np.save(os.path.join(new_dir, save_name), mfcc)


# 获取所有 WAV 文件路径，返回列表
# ['Watkins Marine Mammal Sound (WMMS) Database\\Atlantic_Spotted_Dolphin\\61025001.wav', ...]
audio_files = glob.glob("Watkins Marine Mammal Sound (WMMS) Database/**/*.wav", recursive=True)

'''批量转为 MFCC 文件，并保存在本目录下的 WMMS_MFCC 目录下'''
for wav_path in audio_files:
    save_mfcc(wav_path, "WMMS_MFCC")

print('ok')