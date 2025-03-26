from torchvision import models
from SplitDataset_MSel_ResNet50 import train_dataset, val_dataset
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch

# 检查 GPU 是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用带预训练权重的 resnet50
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# 输出 32 类
model.fc = nn.Linear(model.fc.in_features, 32)
# 冻结部分层
for name, param in model.named_parameters():
    if 'layer3' not in name and 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False
model = model.to(device)

# 使用交叉熵损失函数
loss_fn = nn.CrossEntropyLoss().to(device)
# 设置梯度下降法（Adam）以及学习率 lr
learning_rate = 0.0004
# 设置需要梯度下降的参数
trainable_params = [param for name, param in model.named_parameters() if param.requires_grad]
# Adam 梯度下降法
optimizer = torch.optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)

# ReduceLROnPlateau 动态调整学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # 验证集损失（越小越好），max 表示验证集准确率越大越好
    factor=0.1,          # 学习率乘以 0.1
    patience=5,          # 连续 5 个 epoch 验证集损失未下降时调整学习率
    threshold=0.001,     # 性能改善的最小变化量
    min_lr=1e-6,         # 学习率的最小值
)

# 加载训练集
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# 加载验证集
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# 初始化 TensorBoard
writer = SummaryWriter('logs')

# 训练步数
total_train_step = 0
# 测试步数
total_val_step = 0
# 测试轮次
epoch = 50
# 当前最好准确率（用来保存参数）
best_accuracy = 0.0

for i in range(epoch):
    print("Epoch: {}".format(i+1))

    # 训练模式
    model.train()
    for data in train_dataloader:
        inputs, labels = data # 获得数据
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs) # 前向传播
        loss = loss_fn(output, labels) # 损失函数

        # 反向传播
        optimizer.zero_grad()
        loss.backward() # 梯度计算
        optimizer.step() # 更新参数

        # 记录训练步数
        total_train_step += 1

        if total_train_step % 100 == 0:
            print('Train: {}, loss: {}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step) # 标题 Y X 轴

    # 总测试误差
    total_val_loss = 0
    # 测试集上正确预测数
    total_val_accuracy = 0

    # 测试模式
    model.eval()
    # 关闭梯度计算
    with torch.no_grad():
        for data in val_dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            # 验证集损失
            val_loss = loss_fn(output, labels)
            # 验证集总损失
            total_val_loss += val_loss.item()
            # 验证集正确预测个数
            total_val_accuracy += (output.argmax(1) == labels).sum()

    # 记录测试轮次
    total_val_step += 1

    # 打印测试结果
    print('Validation Loss: {:.4f}'.format(total_val_loss))
    print('Validation Accuracy: {:.2f}%'.format((total_val_accuracy/len(val_dataset))*100))

    # 记录测试结果
    writer.add_scalar('val_loss', total_val_loss, total_val_step)
    writer.add_scalar('val_accuracy', total_val_accuracy/len(val_dataset), total_val_step)

    # 保存最好的模型参数
    current_accuracy = total_val_accuracy/len(val_dataset)
    if (current_accuracy > best_accuracy):
        torch.save(model.state_dict(), "ModelSaved(ResNet50)/resnet50.pth")
        best_accuracy = current_accuracy
        print('Model saved')

    # 学习率衰减，在验证循环中计算平均损失
    scheduler.step(val_loss)

writer.close()