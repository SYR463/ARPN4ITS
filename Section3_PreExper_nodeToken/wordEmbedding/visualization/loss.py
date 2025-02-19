import json
import matplotlib.pyplot as plt


# 从文件加载损失数据
def load_loss_data(file_path):
    with open(file_path, 'r') as file:
        loss_data = json.load(file)
    return loss_data


# 绘制损失变化图
def plot_loss(loss_data, output_path):
    # 获取训练损失和验证损失
    train_loss = loss_data.get("train", [])
    val_loss = loss_data.get("val", [])

    # 绘制训练损失和验证损失
    epochs = range(1, len(train_loss) + 1)  # 假设每个损失值对应一个epoch

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', linestyle='-', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', color='red', linestyle='-', marker='x')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)

    # 保存并显示图像
    plt.savefig(output_path)
    plt.close()  # 关闭图形，以释放内存

if __name__ == '__main__':
    # model_name: cbow, skipgram, combined
    model_name = "combined"
    file_path = f'/mnt/d/project/python/ARPN4ITS/Section3_PreExper_nodeToken/wordEmbedding/weights/{model_name}_preExper/loss.json'
    output_path = f"/mnt/d/project/python/ARPN4ITS/Section3_PreExper_nodeToken/wordEmbedding/visualization/{model_name}_loss.png"

    # 加载并绘制损失变化
    loss_data = load_loss_data(file_path)
    plot_loss(loss_data, output_path)
