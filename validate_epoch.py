import torch
def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()    # 测试模式
    total_loss = 0  # 测试损失
    total_correct = 0   # 测试准确率
    output_list = []
    label_list = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = loss_fn(output, labels)
            # 验证集总损失
            total_loss += loss.item()
            # 验证集正确预测个数
            total_correct += (output.argmax(1) == labels).sum().item()
            output_list.extend(output.argmax(1).cpu().tolist())
            label_list.extend(labels.cpu().tolist())

            output_list.extend(output.argmax(1).cpu().tolist())
            label_list.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader) # 验证集平均损失
    avg_acc = total_correct / len(dataloader.dataset)   # 验证集平均准确率
    return avg_loss, avg_acc, output_list, label_list