import librosa
import numpy as np
import os
import glob


def save_spectrogram(wav_path, save_root, n_fft=512):
    # 加载音频，强制指定采样率 sr 为 16KHz
    y, sr = librosa.load(wav_path, sr=16000)

    # 计算短时傅里叶变换(STFT)的幅度谱
    D = librosa.stft(y, n_fft=n_fft, hop_length=n_fft)  # hop_length 与 n_fft相同
    magnitude = np.abs(D)  # 获取幅度谱

    # 转换为功率谱（幅度平方），使用 np.max 归一化，确保动态范围一致性
    power_spectrum = librosa.amplitude_to_db(magnitude ** 2, ref=np.max)

    # 构建新路径
    rel_path = os.path.relpath(wav_path, start="Watkins Marine Mammal Sound (WMMS) Database")
    new_dir = os.path.join(save_root, os.path.dirname(rel_path))
    os.makedirs(new_dir, exist_ok=True)

    # 保存声谱图为.npy文件
    save_name = os.path.basename(wav_path).replace(".wav", ".npy")
    np.save(os.path.join(new_dir, save_name), power_spectrum)


# 获取所有 WAV 文件路径列表
audio_files = glob.glob("Watkins Marine Mammal Sound (WMMS) Database/**/*.wav", recursive=True)

# 批量转换，保存在本目录下的 WMMS_Spectrogram 目录下
for wav_path in audio_files:
    save_spectrogram(wav_path, "WMMS_Spectrogram")

print('ok-To Spectrogram')