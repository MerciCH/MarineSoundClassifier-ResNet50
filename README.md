# 数据集下载
该模型基于沃特金斯海洋哺乳动物声音数据集训练，数据集下载地址：
[Watkins Marine Mammal Sound Dataset](https://hf-mirror.com/datasets/confit/wmms/tree/main)（只需下载其中的 zip 文件）

请将数据集文件夹放置于项目根目录，整体结构如下：
```
.
├── Watkins Marine Mammal Sound (WMMS) Database
│   └── Atlantic_Spotted_Dolphin
│   │   └── 6102500A.wav
│   │   └── ...
│   └── Bearded_Seal
│   │   └── 7202100T.wav
│   │   └── ...
├── MyDataset_MSel.py
├── SplitDataset_MSel_ResNet50.py
├── Train_MSel(ResNet50).py
├── TransferMelSpectrogram.py
├── TransferMFCC.py
├── TransferSpectrogram.py
```

# 生成特征数据
运行 `TransferMelSpectrogram.py` 会得到梅尔频谱图特征数据，保存为 npy 格式，输出目录为项目根目录，输出文件夹名称为 `WMMS_MelSpectrogram`

运行 `TransferMFCC.py` 得到 MFCC 特征数据，保存为 npy 格式，输出目录为项目根目录，输出文件夹名称为 `WMMS_MFCC`

运行 `TransferSpectrogram.py` 得到声谱图特征数据，保存为 npy 格式，输出目录为项目根目录，输出文件夹名称为 `WMMS_Spectrogram`


# 运行
直接运行 `Train_MSel(ResNet50).py` 即可开始训练；其中训练轮次设置为 50 次，但试验显示训练轮次到 35 次之后，模型预测准确率即可稳定

`ModelSaved(ResNet50)` 文件夹保存的是训练好的模型数据，如果只是想使用模型而非训练模型则可使用该训练权重

**注意**：请确保你的项目根目录有一个 logs 文件夹，没有则新建一个，其中会保存模型训练参数，包括*测试集损失、验证集准确率、验证集损失*

在模型训练结束后，控制台输入 `tensorboard --logdir=logs` 启动数据可视化页面，检测模型训练情况