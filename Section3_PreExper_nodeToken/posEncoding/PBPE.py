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
import time

time = time.strftime("%m%d%H%M", time.localtime())


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

    # angles[::2]：从数组中取偶数索引的元素（对应正弦部分）
    # angles[1::2]：从数组中取奇数索引的元素（对应余弦部分）
    return np.concatenate([np.sin(angles[::2]), np.cos(angles[1::2])])


# 2. Path-Based Positional Encoding（PBPE） - 适用于树结构
def pbpe_encoding(path, max_depth, dim, alpha=1):
    """
    基于路径的位置编码（PBPE）
    :param path: 从根节点到目标节点的路径（如[1, 2, 4]）
    :param max_depth: 树的最大深度
    :param dim: 编码维度
    :param alpha: 层次权重系数（默认值为-0.5）
    :return: 对应路径的PBPE编码
    """
    # 初始化位置编码
    position_enc = np.zeros(dim)
    path_length = len(path)

    # 为每个节点生成正余弦编码，并考虑层次影响
    for i in range(path_length):
        # 计算当前层次的权重：权重为alpha的i次方
        layer_weight = alpha ** i  # 例如第一层为alpha^0，第二层为alpha^1，依此类推

        # 为节点生成正余弦编码
        sin_cos_encoding = sinusoidal_encoding(path[i], dim // path_length)

        # 调整每个节点的编码以考虑其层次权重
        position_enc[i * (dim // path_length):(i + 1) * (dim // path_length)] = layer_weight * sin_cos_encoding

    # # 根节点处理：可以增强根节点的影响（根据需求可调整）
    # if path_length == 1:  # 根节点
    #     position_enc *= 2  # 假设增强根节点的影响

    return position_enc


def pbpe_encoding_with_weight(path, max_depth, dim, alpha=-0.5):
    """
    基于路径的位置编码（PBPE）
    :param path: 从根节点到目标节点的路径（如[1, 2, 4]）
    :param max_depth: 树的最大深度
    :param dim: 编码维度
    :param alpha: 层次权重系数（默认值为-0.5）
    :return: 对应路径的PBPE编码
    """
    # 初始化路径编码
    position_enc = np.zeros(dim)
    path_length = len(path)

    # 为每个节点生成正余弦编码，并将它们拼接成路径编码
    for i in range(path_length):
        sin_cos_encoding = sinusoidal_encoding(path[i], dim // path_length)  # 为节点生成正余弦编码
        position_enc[i * (dim // path_length):(i + 1) * (dim // path_length)] = sin_cos_encoding

    # 计算层次权重：对整个路径编码乘以层次权重（alpha的path_length次方）
    layer_weight = alpha ** (path_length - 1)  # 计算路径的层次权重：假设路径长度为3，根节点的权重为1，第二层为alpha，第三层为alpha^2
    position_enc = position_enc * layer_weight # 对整个路径编码乘以层次权重

    # # 根节点处理：可以增强根节点的影响（根据需求可调整）
    # if path_length == 1:  # 根节点
    #     path_encoding *= 2  # 假设增强根节点的影响

    return position_enc


# 3. 计算树中每个节点的位置编码
def calculate_node_encodings(tree_structure, max_depth, dim, alpha=-0.5):
    """
    计算树中每个节点的位置编码
    :param tree_structure: 树的结构，节点的路径信息（如[[8], [8, 6], [8, 7], [8, 6, 1], [8, 6, 2], [8, 7, 1], [8, 7, 2], [8, 7, 3]]）
    :param dim: 编码维度
    :return: 所有节点的位置编码
    """
    node_encodings = []

    # 遍历树的每一个节点
    for path in tree_structure:
        encoding = pbpe_encoding(path, max_depth, dim, alpha)
        node_encodings.append(encoding)

    return np.array(node_encodings)


# 3. 提取树结构中的位置编码（去除节点信息）
def extract_position_encoding():
    # 树结构的路径（如路径从根节点到叶节点）
    max_depth = 3  # 树的最大深度为2
    tree_paths = [
        [8],  # 根节点
        [8, 6],
        [8, 7],
        [8, 6, 1],
        [8, 6, 2],
        [8, 7, 1],
        [8, 7, 2],
        [8, 7, 3],
    ]  # 树的路径

    # 提取每个路径的PBPE编码（去除节点信息）
    pbpe_codes = np.array([pbpe_encoding(path, max_depth, 16, alpha=1) for path in tree_paths])  # 假设编码维度为16

    # 确保 pbpe_codes 统一为相同的形状
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
    ax.set_yticklabels(['R8', 'R6', 'R7', 'R1', 'R2', 'R3', 'R4', 'R5'])  # 设置y轴标签为节点ID

    # 在每个格子中显示其对应的值
    for i in range(len(pbpe_codes)):
        for j in range(pbpe_codes.shape[1]):
            value = round(pbpe_codes[i, j], 2)  # 取小数点后两位
            ax.text(j, i, str(value), ha="center", va="center", color="white", fontsize=10)

    # 添加色条
    fig.colorbar(cax)

    # 保存图像到文件
    plt.tight_layout()
    plt.savefig(f'PBPE_posEnc_{time}.png')
    print("图像已保存为 'PBPE_position_encoding.png'")


if __name__ == '__main__':
    # 运行并可视化
    visualize_position_encoding()

if __name__ == '__main1__':
    # 示例：树结构
    tree_structure = {
        '4': ['41', '42'],
        '41': ['411', '412', '413'],
        '42': ['421', '422', '423']
    }

    # 树的最大深度（假设树的最大深度为3）
    max_depth = 3

    # 示例：对路径 [4, 41, 411] 进行编码
    path = [4, 41, 411]
    dim = 6  # 编码维度

    # 获取对应路径的PBPE编码
    encoded_position = pbpe_encoding(path, max_depth, dim)

    print(f"Encoded Position for path {path}:")
    print(encoded_position)
