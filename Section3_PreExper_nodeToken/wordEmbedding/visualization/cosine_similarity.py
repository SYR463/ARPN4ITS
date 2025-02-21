import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import os
import json
import time

# 设置参数, model_name: cbow, skipgram, combined
model_name = "combined"
tree_num = 1

time = time.strftime("%m%d%H%M", time.localtime())
vocab_path = "/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json"
model_dir = f"/mnt/d/project/python/ARPN4ITS/Section3_PreExper_nodeToken/wordEmbedding/weights/{model_name}_preExper"
pic_title = f"Cosine Similarity of R-tree_{tree_num}"
output_path = f"cosSim/cosSim_{model_name}_tree{tree_num}_{time}.png"


# 可视化余弦相似度矩阵
def plot_cosine_similarity_matrix(similarity_matrix, word_list, custom_label_mapping,
                                  title=pic_title, output_path=output_path):
    """
    绘制余弦相似度矩阵
    :param similarity_matrix: 余弦相似度矩阵
    :param word_list: 词汇列表
    :param custom_labels: 自定义标签列表
    :param title: 图标题
    """
    plt.figure(figsize=(12, 8))

    # 创建自定义标签列表，使用映射字典来获取每个节点的简化标签
    custom_labels = [custom_label_mapping.get(node, node) for node in word_list]  # 映射节点到简化标签

    # 绘制热图
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu", annot_kws={"size": 14},
                xticklabels=custom_labels, yticklabels=custom_labels)
    plt.title(title)
    # plt.xlabel("Words")
    # plt.ylabel("Words")

    # 设置标签的字体大小
    plt.xticks(rotation=0, fontsize=16)  # 调整x轴标签字体大小并旋转标签
    plt.yticks(rotation=0, fontsize=16)  # 调整y轴标签字体大小

    # 保存并显示图像
    plt.savefig(output_path)
    plt.close()  # 关闭图形，以释放内存


# 加载词汇表
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as file:
        vocab = json.load(file)
    return vocab


# 获取词汇的嵌入并计算每个节点与所有节点的相似度（包括自身）
def visualize_word_embeddings_from_saved(model_dir, tree_nodes, custom_label_mapping):
    """
    从保存的模型中加载词向量并计算每个节点与所有节点的余弦相似度
    :param model_dir: 模型目录
    :param tree_nodes: 树节点列表
    :param custom_label_mapping: 自定义映射规则
    """
    # 加载保存的词嵌入和词汇表
    embeddings = np.load(os.path.join(model_dir, "word_embeddings.npy"))
    word_list = load_vocab(vocab_path)  # 获取模型的词汇列表
    word_list = list(word_list.keys())  # 取所有词汇，不限制数量

    similarity_matrix = []  # 用于存储所有目标节点与上下文节点之间的相似度矩阵

    # 遍历每个节点，计算该节点与所有其他节点的相似度，包括自身
    for target_node in tree_nodes:
        if target_node not in word_list:
            print(f"目标节点 {target_node} 不在词汇表中！")
            continue

        # 获取目标节点的嵌入
        target_index = word_list.index(target_node)
        target_embedding = embeddings[target_index]

        # 获取所有上下文节点（包括目标节点自身）
        context_embeddings = []
        for context_node in tree_nodes:
            if context_node in word_list:
                context_index = word_list.index(context_node)
                context_embeddings.append(embeddings[context_index])

        # 计算目标节点与每个上下文节点的余弦相似度（包括自身）
        similarity_scores = []
        for context_embedding in context_embeddings:
            similarity_scores.append(cosine_similarity([target_embedding], [context_embedding])[0][0])

        # 输出相似度
        print(f"目标节点 {target_node} 与上下文节点的相似度:")
        for context_node, similarity_score in zip(tree_nodes, similarity_scores):
            print(f"目标节点 {target_node} 和 上下文节点 {context_node} 之间的余弦相似度: {similarity_score:.4f}")

        # 将当前目标节点的相似度添加到矩阵中
        similarity_matrix.append(similarity_scores)

    # 将相似度矩阵转换为NumPy数组并可视化
    similarity_matrix = np.array(similarity_matrix)
    plot_cosine_similarity_matrix(similarity_matrix, tree_nodes, custom_label_mapping)


if __name__ == '__main__':
    """
    example1: 
    <NL>A0E5
      <NL>A3E5
        <L>A3B4
        <L>D4E5
      <NL>B0E3
        <L>B1C2
        <L>C2D3
        <L>D0E1
    """
    tree_nodes1 = ["<NL>A0E5", "<NL>A3E5", "<L>A3B4", "<L>D4E5",
                    "<NL>B0E3", "<L>B1C2", "<L>C2D3", "<L>D0E1"]
    # 自定义标签映射字典
    custom_label_mapping1 = {
        "<NL>A0E5": "R8",
        "<NL>A3E5": "R6",
        "<L>A3B4": "R2",
        "<L>D4E5": "R1",
        "<NL>B0E3": "R7",
        "<L>B1C2": "R4",
        "<L>C2D3": "R3",
        "<L>D0E1": "R5"
    }

    """
    example2:
    <NL>A1E5
      <NL>A4E5
        <L>A4B5
        <L>D4E5
      <NL>B1E3
        <L>B2C3
        <L>C2D3
        <L>D1E2
    """
    tree_nodes2 = ["<NL>A1E5", "<NL>A4E5", "<L>A4B5", "<L>D4E5",
                    "<NL>B1E3", "<L>B2C3", "<L>C2D3", "<L>D1E2"]
    # 自定义标签映射字典
    custom_label_mapping2 = {
        "<NL>A1E5": "R8",
        "<NL>A4E5": "R6",
        "<L>A4B5": "R2",
        "<L>D4E5": "R1",
        "<NL>B1E3": "R7",
        "<L>B2C3": "R4",
        "<L>C2D3": "R3",
        "<L>D1E2": "R5"
    }

    # 从保存的模型中加载词向量并进行每个节点与上下文节点的余弦相似度计算
    if tree_num == 1:
        visualize_word_embeddings_from_saved(model_dir, tree_nodes1, custom_label_mapping1)
    else:
        visualize_word_embeddings_from_saved(model_dir, tree_nodes2, custom_label_mapping2)


if __name__ == '__main1__':
    """
    "<NL>A2E5", "<NL>B4D5", "<L>B4C5", "<L>C4D5", "<NL>A2C3", "<L>A2B3", "<L>B2C3", "<NL>B3E4", "<L>B3C4", "<L>D3E4" 
    """
    tree_nodes = ["<NL>A2E5", "<NL>B4D5", "<L>B4C5", "<L>C4D5",
                  "<NL>A2C3", "<L>A2B3", "<L>B2C3",
                  "<NL>B3E4", "<L>B3C4", "<L>D3E4"]
    # 自定义标签映射字典
    custom_label_mapping = {
        "<NL>A2E5": "R10",
        "<NL>B4D5": "R9",
        "<L>B4C5": "R4",
        "<L>C4D5": "R6",
        "<NL>A2C3": "R8",
        "<L>A2B3": "R5",
        "<L>B2C3": "R1",
        "<NL>B3E4": "R7",
        "<L>B3C4": "R2",
        "<L>D3E4": "R3"
    }

    visualize_word_embeddings_from_saved(model_dir, tree_nodes, custom_label_mapping)
