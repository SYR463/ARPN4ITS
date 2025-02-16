"""
树结构：

 * 4 <NL> A0 E5
 * 41 <NL> A0 C5
 * 411 <L> B0 C1
 * 412 <L> B3 C4
 * 413 <L> A4 B5
 * 42 <NL> D2 E5
 * 421 <L> D2 E3
 * 422 <L> D2 E3
 * 423 <L> D4 E5
"""

import numpy as np
import matplotlib.pyplot as plt


# 1. 正余弦编码（Sinusoidal Encoding） - 适用于序列结构
def sinusoidal_encoding(position, dim):
    """
    生成正余弦位置编码
    :param position: 位置
    :param dim: 嵌入维度
    :return: 对应位置的正余弦编码
    """
    angle_rates = 1 / np.power(10000, (2 * (np.arange(dim) // 2)) / np.float32(dim))  # 计算角度比率
    angles = position * angle_rates  # 计算每个位置的角度
    return np.concatenate([np.sin(angles[::2]), np.cos(angles[1::2])])


# 2. Path-Based Positional Encoding（PBPE） - 适用于树结构
def pbpe_encoding(path, max_depth, dim):
    """
    基于路径的位置编码（PBPE）
    :param path: 从根节点到目标节点的路径（如[1, 2, 4]）
    :param max_depth: 树的最大深度
    :param dim: 编码维度
    :return: 对应路径的PBPE编码
    """
    # 基于路径的编码和位置的正余弦编码结合
    position_enc = np.zeros(dim)
    path_length = len(path)
    for i in range(path_length):
        # 路径中的节点使用位置编码
        position_enc[i] = (path[i] + 1) / (max_depth + 1)

    # 使用正余弦编码来表示位置
    return sinusoidal_encoding(position_enc, dim)


# 3. 提取树结构中的位置编码（去除节点信息）
def extract_position_encoding():
    # 树结构的路径（如路径从根节点到叶节点）
    max_depth = 3  # 树的最大深度为2
    tree_paths = [
        [8],    # 根节点
        [8, 6],
        [8, 7],
        [8, 6, 1],
        [8, 6, 2],
        [8, 7, 1],
        [8, 7, 2],
        [8, 7, 3],
    ]  # 树的路径

    # 提取每个路径的PBPE编码（去除节点信息）
    pbpe_codes = np.array([pbpe_encoding(path, max_depth, 16) for path in tree_paths])  # 假设编码维度为16

    return pbpe_codes


# 4. 绘制编码结果
# 绘制编码结果
def visualize_position_encoding():
    # 提取树结构中的位置编码
    pbpe_codes = extract_position_encoding()

    # 可视化位置编码
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(pbpe_codes, cmap='viridis', aspect='auto')
    ax.set_title('Path-Based Positional Encoding (PBPE) for Tree Structure')
    ax.set_ylabel('Node Path')
    ax.set_xlabel('Dimension')

    # 设置y轴标签为节点ID
    ax.set_yticks(np.arange(len(pbpe_codes)))  # 设置y轴位置
    # ax.set_yticklabels([f'Node {i+1}' for i in range(len(pbpe_codes))])  # 设置y轴标签为节点ID
    ax.set_yticklabels([8, 6, 7, 1, 2, 3, 4, 5])  # 设置y轴标签为节点ID

    # 在每个格子中显示其对应的值
    for i in range(len(pbpe_codes)):
        for j in range(pbpe_codes.shape[1]):
            value = round(pbpe_codes[i, j], 2)  # 取小数点后两位
            ax.text(j, i, str(value), ha="center", va="center", color="white", fontsize=8)


    # 添加色条
    fig.colorbar(cax)

    # 保存图像到文件
    plt.tight_layout()
    plt.savefig('PBPE_position_encoding.png')
    print("图像已保存为 'PBPE_position_encoding.png'")



if __name__ == '__main__':

    # 运行并可视化
    visualize_position_encoding()
