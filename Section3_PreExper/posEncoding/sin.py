import numpy as np
import matplotlib.pyplot as plt


# 先序遍历树结构并生成节点序列
def preorder_traversal(root):
    result = []
    if root:
        result.append(root['value'])  # 添加当前节点
        for child in root['children']:
            result.extend(preorder_traversal(child))  # 递归遍历子节点
    return result


# 正余弦编码函数
def sinusoidal_encoding(position, dim):
    """
    为给定位置生成正余弦位置编码
    :param position: 当前节点在序列中的位置
    :param dim: 编码的维度
    :return: 对应位置的正余弦编码
    """
    angle_rates = 1 / np.power(10000, (2 * (np.arange(dim) // 2)) / np.float32(dim))  # 计算角度比率
    angles = position * angle_rates  # 计算每个位置的角度
    return np.concatenate([np.sin(angles[::2]), np.cos(angles[1::2])])  # 奇数维度使用正弦，偶数维度使用余弦




# 4. 绘制编码结果
def visualize_position_encoding(encoded_positions, node_sequence):
    # 提取树结构中的位置编码
    encoded_positions = np.array(encoded_positions)

    # 可视化位置编码
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(encoded_positions, cmap='viridis', aspect='auto')
    ax.set_title('Positional Encoding for Serialized Tree Structure')
    ax.set_ylabel('Node Position in Sequence')
    ax.set_xlabel('Dimension')

    # 设置y轴标签为节点ID
    ax.set_yticks(np.arange(len(node_sequence)))  # 设置y轴位置
    ax.set_yticklabels(node_sequence)  # 设置y轴标签为节点ID

    # 在每个格子中显示其对应的值
    for i in range(len(node_sequence)):
        for j in range(encoded_positions.shape[1]):
            value = round(encoded_positions[i, j], 2)  # 取小数点后两位
            ax.text(j, i, str(value), ha="center", va="center", color="white", fontsize=8)

    # 添加色条
    fig.colorbar(cax)

    # 保存图像到文件
    plt.tight_layout()
    plt.savefig('sin_position_encoding.png')
    print("图像已保存为 'sin_position_encoding.png'")


if __name__ == '__main__':
    # 创建树结构示例
    # tree = {
    #     'value': 1,
    #     'children': [
    #         {'value': 2, 'children': [{'value': 4, 'children': []}, {'value': 5, 'children': []}]},
    #         {'value': 3, 'children': [{'value': 6, 'children': []}]}
    #     ]
    # }

    # 序列化树结构
    # node_sequence = preorder_traversal(tree)
    node_sequence = [8, 6, 1, 2, 7, 3, 4, 5]
    print(f"序列化后的树结构：{node_sequence}")

    # 对序列中的每个节点生成正余弦编码
    dim = 16  # 设置编码的维度
    encoded_positions = [sinusoidal_encoding(i, dim) for i in range(len(node_sequence))]

    # 可视化树结构的正余弦编码
    visualize_position_encoding(encoded_positions, node_sequence)
