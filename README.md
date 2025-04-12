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
直接运行 `Train_MSel(ResNet50).py` 即可开始训练；其中训练轮次设置为 50 次，但试验显示训练轮次到 35 次之后，模型预测准确率即可稳定（我加入了对 VGG16 网络的训练代码，该网络测试效果同样十分优秀）

**注意**：请确保你的项目根目录有 `logs`、`CM_MSel_ResNet50`、`CM_MSel_VGG16（若使用 VGG16 训练代码）`、`ModelSaved_MSel(ResNet50)`、`ModelSaved_MSel(VGG16)（若使用 VGG16 训练代码）` 五个文件夹，没有则新建，其中 logs 目录会保存模型训练参数，包括*学习率衰减情况、模型准确率、召回率、F1分数等内容*
CM 开头的目录会保存模型训练在训练阶段成绩较好的混淆矩阵，ModelSaved 目录则会保存最好的模型参数

在模型训练结束后，控制台输入 `tensorboard --logdir=logs` 启动数据可视化页面，检测模型训练情况

# 模型性能
## 正确率、F1分数、精确率、召回率
![性能分数](https://github.com/user-attachments/assets/54ede33b-4995-4caa-8dbf-bfbf6a1f7b59)

## 混淆矩阵
![最佳混淆矩阵](https://github.com/user-attachments/assets/95bf5c11-b217-4ba2-be11-dd99512b0e3c)

