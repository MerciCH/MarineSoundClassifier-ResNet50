def train_epoch(model, dataloader, loss_fn, optimizer, device, writer, total_train_step):
    model.train()   # 训练模式
    total_loss = 0  # 训练集损失
    total_correct = 0   #训练集准确率

    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)  # 前向传播
        loss = loss_fn(output, labels)  # 损失计算

        # 反向传播
        optimizer.zero_grad()
        loss.backward()  # 梯度计算
        optimizer.step()  # 更新参数

        total_train_step += 1  # 训练步数
        total_loss += loss.item()  # 训练集总损失
        total_correct += (output.argmax(1) == labels).sum().item()  # 训练集正确预测数

        if total_train_step % 100 == 0:
            print(f"Train: {total_train_step}, loss: {loss.item():.4f}")
            writer.add_scalar('train_loss_per_step', loss.item(), total_train_step)  # 每 100 步的训练损失

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc