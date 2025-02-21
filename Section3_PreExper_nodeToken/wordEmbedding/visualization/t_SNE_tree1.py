import json
import os
import random
import re

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import time
from adjustText import adjust_text

from Section3_PreExper_nodeToken.wordEmbedding.utils.dataloader import load_vocab
from Section3_PreExper_nodeToken.wordEmbedding.constructContext import parse_tree, print_node_info, print_node_context_info
from Section3_PreExper_nodeToken.wordEmbedding.visualization.t_SNE import extract_word_embeddings, tsne_embeddings, get_embeddings_for_word_list
from Section3_PreExper_nodeToken.wordEmbedding.filterByNode.wordCount import get_word_list

# model_name: cbow, skipgram, combined
model_name = "skipgram"

time = time.strftime("%m%d%H%M", time.localtime())
vocab_path = "/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json"
tsne_output_path = f"tsne/tsne_{model_name}_tree_{time}.png"
model_dir = f"/mnt/d/project/python/ARPN4ITS/Section3_PreExper_nodeToken/wordEmbedding/weights/{model_name}_preExper"
pic_title = f"t-SNE of {model_name}"

dataset_path = "/mnt/d/project/Java/rtree_construct/preExperData/RTreeToken"



def plot_tsne_embeddings(reduced_embeddings, word_list, colors, title=pic_title, output_path=tsne_output_path):
    """
    绘制t-SNE降维后的词向量
    :param reduced_embeddings: t-SNE降维后的二维词向量
    :param word_list: 词汇列表
    :param colors: 每个词汇对应的颜色
    :param title: 图标题
    """
    plt.figure(figsize=(12, 8))

    # 绘制散点图
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, s=50, alpha=0.7)

    # 添加词汇标签
    texts = []
    for i, word in enumerate(word_list):
        text = plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], word, fontsize=5, alpha=0.7, ha='center')
        texts.append(text)

    # 自动调整标签位置，避免重叠
    adjust_text(texts)

    plt.title(title)

    plt.savefig(output_path, dpi=600)
    plt.close()  # 关闭图形，以释放内存


import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

def define_token_colors(word_list, parsed_tree):
    """
    根据词汇列表和树结构生成颜色列表，确保树的不同分支使用不同的颜色
    :param word_list: 词汇列表
    :param parsed_tree: 已解析的树结构，包含节点及其子节点信息
    :return: 每个词汇对应的颜色列表
    """
    colors = []
    color_map = {}

    # 给每个分支分配不同的颜色
    def assign_color(node, color):
        # 使用节点的 "line" 或 "id" 作为唯一标识符
        node_id = node.get("line", str(node))  # 假设每个节点有 "line" 字段
        if node_id not in color_map:
            color_map[node_id] = color

        # 获取当前节点的子节点并递归分配颜色
        for child in node.get("children", []):
            assign_color(child, color)

    # 随机生成不同分支的颜色
    color_list = plt.cm.tab20.colors  # 使用 tab20 色系来区分不同分支
    idx = 0

    # 处理根节点组
    root_node = parsed_tree[0]  # 假设 parsed_tree 的第一个节点是根节点
    root_color = color_list[idx % len(color_list)]
    assign_color(root_node, root_color)
    idx += 1

    # 处理第一层子节点组
    for child in root_node.get("children", []):
        child_color = color_list[idx % len(color_list)]
        assign_color(child, child_color)
        idx += 1

    # 处理第二层子节点组（子节点的孩子节点）
    for child in root_node.get("children", []):
        for grandchild in child.get("children", []):
            grandchild_color = color_map.get(child.get("line", str(child)))  # 继承父节点颜色
            color_map[grandchild.get("line", str(grandchild))] = grandchild_color

    # 将颜色分配到每个词汇
    for word in word_list:
        color = color_map.get(word, (217 / 255, 217 / 255, 217 / 255))  # 默认灰色
        colors.append(color)

    return colors



if __name__ == '__main__':

    # 测试代码：调用 parse_tree 函数来解析树并为词汇分配颜色
    text = """
        <NL>A0E5
          <NL>A3E5
            <L>A3B4
            <L>D4E5
          <NL>B0E3
            <L>B1C2
            <L>C2D3
            <L>D0E1
    """
    parsed_tree = parse_tree(text)
    word_list = ["<NL>A0E5", "<NL>A3E5", "<L>A3B4", "<NL>B0E3", "Subchild 2"]
    colors = define_token_colors(word_list, parsed_tree)

    # 打印分配的颜色
    for word, color in zip(word_list, colors):
        print(f"Word: {word}, Color: {color}")

def define_token_colors1(word_list, parsed_tree):
    """
    根据词汇列表和树结构生成颜色列表，确保树的不同分支使用不同的颜色
    :param word_list: 词汇列表
    :param tree_structure: 树的父子关系
    :return: 每个词汇对应的颜色列表
    """
    colors = []
    color_map = {}

    # 给每个分支分配不同的颜色
    def assign_color(node, color):
        # 假设节点包含一个 'text' 或 'id' 字段作为唯一标识
        node_id = node.get("text", node.get("id", str(node)))  # 使用 'text' 或 'id' 作为唯一标识符
        if node_id not in color_map:
            color_map[node_id] = color

        # 如果子节点存在并且是列表类型
        children = node.get("children", [])
        for child in children:
            assign_color(child, color)

    # 随机生成不同分支的颜色
    color_list = plt.cm.tab20.colors  # 使用 tab20 色系来区分不同分支
    idx = 0
    for parent in parsed_tree:
        color = color_list[idx % len(color_list)]  # 根据树的结构分配颜色
        assign_color(parent, color)
        idx += 1

    # 将颜色分配到每个词汇
    for word in word_list:
        color = color_map.get(word, (217 / 255, 217 / 255, 217 / 255))  # 如果没有颜色，默认使用灰色
        colors.append(color)

    return colors


def visualize_word_embeddings_from_saved(model_dir, folder_path, parsed_tree, num_words=1000):
    # 获取前 num_words 个频次最高的词汇和 JSON 字符串
    sorted_token_frequencies, _ = get_word_list(folder_path = dataset_path, top_n=num_words)
    # 提取前 num_words 个词汇
    word_list = [token for token, freq in sorted_token_frequencies]

    # 加载保存的词嵌入和词汇表
    embeddings = np.load(os.path.join(model_dir, "word_embeddings.npy"))
    vocab = load_vocab(vocab_path)  # 获取模型的词汇列表

    # 获取这些词汇对应的嵌入
    used_embeddings = get_embeddings_for_word_list(word_list, vocab, embeddings)

    # 对词向量进行t-SNE降维
    reduced_embeddings = tsne_embeddings(used_embeddings)

    # 可视化t-SNE结果
    colors = define_token_colors(word_list, parsed_tree)
    plot_tsne_embeddings(reduced_embeddings, word_list, colors=colors)



# 读取文件内容
def read_file(input_path):
    if not input_path:
        raise ValueError("The input file path cannot be empty.")  # 检查输入路径是否为空

    # 读取文件内容
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


if __name__ == '__main1__':
    file_path = "/mnt/d/project/Java/rtree_construct/preExperData/testDataset/RTreeToken/tokens_output_1118846982100_RTree.txt"

    text = read_file(file_path)
    parsed_tree = parse_tree(text)
    # # 查找并打印某一节点的上下文信息
    # for node in parsed_tree:
    #     # 这里示范打印每个节点的信息，可以指定某个节点来查看
    #     res = print_node_info(node)
    #     print(res)

    visualize_word_embeddings_from_saved(model_dir, file_path, parsed_tree = parsed_tree, num_words=1000)
