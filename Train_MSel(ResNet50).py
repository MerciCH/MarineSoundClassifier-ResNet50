from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from SplitDataset_MSel import train_dataset, val_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import seaborn as sns
import time

from create_resnet50 import create_model
from train_epoch import train_epoch
from validate_epoch import validate_epoch

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model(device)

# 统计模型参数
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# 使用交叉熵损失函数
loss_fn = nn.CrossEntropyLoss().to(device)
# 设置梯度下降法（Adam）以及学习率 lr
learning_rate = 0.0004
# 设置需要梯度下降的参数
trainable_params = [param for name, param in model.named_parameters() if param.requires_grad]
# Adam 梯度下降法（添加权重衰减防止过拟合）
optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)

# ReduceLROnPlateau 动态调整学习率（根据验证集损失调整）
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # 验证集损失（越小越好），max 表示验证集准确率越大越好
    factor=0.1,          # 学习率乘以 0.1
    patience=5,          # 连续 5 个 epoch 验证集损失未下降时调整学习率
    threshold=0.001,     # 性能改善的最小变化量
    min_lr=1e-6,         # 学习率的最小值
)

# 加载数据集
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 初始化 TensorBoard
writer = SummaryWriter('logs')

# 训练步数
total_train_step = 0
# 测试步数
total_val_step = 0
# 总训练轮次
epoch = 50
# 当前最好准确率（用来保存参数）
best_accuracy = 0.0

# 记录训练开始时间
start_time = time.time()

for i in range(epoch):
    print(f"\nEpoch: {i+1}")

    # 训练开始时间
    epoch_start_time = time.time()

    avg_train_loss, avg_train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device, writer,
                                                total_train_step)
    avg_val_loss, avg_val_acc, output_list, label_list = validate_epoch(model, val_dataloader, loss_fn, device)

    print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc * 100:.2f}%")
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_acc * 100:.2f}%")
    writer.add_scalar('train_loss', avg_train_loss, i + 1)
    writer.add_scalar('train_accuracy', avg_train_acc, i + 1)
    writer.add_scalar('val_loss', avg_val_loss, i + 1)
    writer.add_scalar('val_accuracy', avg_val_acc, i + 1)

    # 学习率
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('learning_rate', current_lr, i + 1)

    # 每个 epoch 的训练时间
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    print(f"Epoch {i + 1} took {epoch_time:.2f} seconds")
    writer.add_scalar('epoch_time', epoch_time, i + 1)

    # 准确率、召回率、F1分数
    num_classes = 32
    report = classification_report(
        label_list,
        output_list,
        labels=range(num_classes),  # 强制包含所有类别
        output_dict=True,
        zero_division=0  # 处理分母为0的情况
    )

    # 总体指标
    overall_metrics = report['weighted avg']
    accuracy = report['accuracy']
    precision = overall_metrics['precision']
    recall = overall_metrics['recall']
    f1_score = overall_metrics['f1-score']

    # 记录总体指标
    writer.add_scalars(
        'overall_metrics',
        {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        },
        i + 1
    )

    # 保存最好的模型参数
    if avg_val_acc > best_accuracy:
        best_accuracy = avg_val_acc
        torch.save(model.state_dict(), "ModelSaved_MSel(ResNet50)/resnet50.pth")
        # 保存最佳 epoch 的混淆矩阵
        cm = confusion_matrix(label_list, output_list)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Best Epoch {i + 1} Confusion Matrix')
        plt.savefig(f'CM_MSel_ResNet50/bestEpoch_{i + 1}.png')
        plt.close()
        print('Model and ConfusionMatrix saved')

    # 学习率调整（使用验证集平均损失）
    scheduler.step(avg_val_loss)

# 记录总训练时间
total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")
writer.add_text('total_time', f"Total training time: {total_time:.2f} seconds")

writer.close()