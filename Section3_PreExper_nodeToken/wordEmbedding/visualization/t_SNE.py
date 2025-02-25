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
from Section3_PreExper_nodeToken.wordEmbedding.filterByNode.wordCount import get_word_list

# model_name: cbow, skipgram, combined
model_name = "combined"

time = time.strftime("%m%d%H%M", time.localtime())
vocab_path = "/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json"
tsne_output_path = f"tsne/tsne_{model_name}_{time}.png"
model_dir = f"/mnt/d/project/python/ARPN4ITS/Section3_PreExper_nodeToken/wordEmbedding/weights/{model_name}_preExper"
pic_title = f"t-SNE of {model_name}"

dataset_path = "/mnt/d/project/Java/rtree_construct/preExperData/RTreeToken"

def extract_word_embeddings(model):
    """
    提取 Skip-Gram 模型中的词嵌入
    :param model: 训练好的 Skip-Gram 模型
    :return: 词嵌入矩阵 (word embeddings)
    """
    embeddings = model.embeddings.weight.detach().cpu().numpy()
    return embeddings


def tsne_embeddings(embeddings, n_components=2, random_state=42):
    """
    使用t-SNE将高维词嵌入降维到2D
    :param embeddings: 词嵌入矩阵
    :param n_components: 降维后的维度，默认为2
    :param random_state: 随机种子
    :return: 降维后的词向量
    """
    tsne = TSNE(n_components=n_components, random_state=random_state)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def plot_tsne_embeddings(reduced_embeddings, word_list, colors, title=pic_title, output_path=tsne_output_path):
    """
    绘制t-SNE降维后的词向量
    :param reduced_embeddings: t-SNE降维后的二维词向量
    :param word_list: 词汇列表
    :param title: 图标题
    """
    plt.figure(figsize=(12, 8))

    # 绘制散点图
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, s=50, alpha=0.7)
    # plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=50, alpha=0.7)

    # 添加词汇标签
    texts = []
    for i, word in enumerate(word_list):
        text = plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], word, fontsize=5, alpha=0.7, ha='center')
        texts.append(text)

    # # 添加标签（只为深棕色和墨绿色token显示标签）
    # for i, word in enumerate(word_list):
    #     if word in deep_brown_tokens or word in dark_green_tokens:
    #         plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=10, alpha=0.7)

    # 使用adjustText自动调整标签位置，避免重叠
    # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    adjust_text(texts)


    plt.title(title)

    # # 设置x轴和y轴的显示范围，调整这部分以只聚焦某一区域
    # plt.xlim([-1.5, -1.15])  # 调整为适合的x轴范围
    # plt.ylim([-2.1, -1.6])  # 调整为适合的y轴范围

    plt.savefig(output_path, dpi=600)
    plt.close()  # 关闭图形，以释放内存



def define_token_colors(word_list):
    """
    根据词汇列表和预定义的token集合生成颜色
    :param word_list: 词汇列表
    :return: 每个词汇对应的颜色列表
    """
    # # 定义token集合
    # black_tokens = {"<NL>A0E5"}
    # deep_brown_tokens = {"<NL>A3E5", "<L>D4E5", "<L>A3B4"}  # 深棕色token集合
    # dark_green_tokens = {"<NL>B0E3", "<L>C2D3", "<L>B1C2", "<L>D0E1"}  # 墨绿色token集合

    # 为每个词汇分配颜色
    colors = []

    # 定义正则表达式匹配“<NL>A_B_”类型的token
    # pattern = r"<NL>A_[0-9]_B_[0-9]"

    # pattern = r"^<NL>.*"
    # pattern = r"^<NL>A\dB\d$"

    # 定义正则表达式匹配以"<NL>"开头的token，后跟字母和数字
    pattern = r"^<NL>([A-Z])\d([A-Z])\d$"  # 匹配模式：<NL>A_B_、<NL>A_C_ 等
    # pattern = r"^<NL>([A-Z])([0-9])([A-Z])([0-9])"  # 匹配模式：<NL>A_B_、<NL>A_C_ 等
    # pattern = r"^<L>.*"  # 匹配模式：<L>*
    # pattern = r"^<NL>([A-Z]).*"  # 匹配模式：<NL>A_、<NL>B_ 等


    # 使用matplotlib自带的tab20色系
    color_map = {}
    tab20_colors = plt.cm.tab20.colors  # 获取tab20色系的颜色

    # for word in word_list:
    #     if word in deep_brown_tokens:
    #         colors.append((197 / 255, 90 / 255, 17 / 255))  # 深棕色
    #     elif word in dark_green_tokens:
    #         colors.append((0 / 255, 188 / 255, 18 / 255))  # 墨绿色
    #     elif word in black_tokens:
    #         colors.append('black')
    #     else:
    #         colors.append((180 / 255, 199 / 255, 231 / 255))  # 浅蓝色

    # 高亮非叶节点
    for word in word_list:
        match = re.match(pattern, word)
        if match:
            # 按照横坐标分配颜色, 即AB, AC
            rule = match.group(1) + match.group(2)
            # rule = match.group(1)

            # 按照面积分配颜色面积
            # rule = ord(match.group(3)) - ord(match.group(1)) + int(match.group(4)) - int(match.group(2))

            # 随机生成颜色并记录，确保颜色之间的差异较大
            if rule not in color_map:
                color_map[rule] = tab20_colors[len(color_map) % len(tab20_colors) * 2]

            # 为该规则的token分配颜色
            colors.append(color_map[rule])
        else:
            colors.append((217/255, 217/255, 217/255))  # 灰色，其余叶节点

    # # 高亮叶节点
    # for word in word_list:
    #     if re.match(pattern, word):
    #         colors.append(tab20_colors[2])
    #     else:
    #         colors.append((217/255, 217/255, 217/255))  # 灰色，其余叶节点

    return colors


def get_embeddings_for_word_list(word_list, vocab, embeddings):
    """
    获取词汇表中实际使用的词汇的词向量。
    :param word_list: 词汇列表（从高频排序后的词汇）
    :param vocab: 词汇表，字典格式，键是词，值是词的索引
    :param embeddings: 词嵌入矩阵
    :return: 词向量
    """
    # 根据 word_list 中的词汇，从词汇表中获取对应的索引
    word_indices = [vocab[word] for word in word_list if word in vocab]

    # 获取对应的词嵌入
    used_embeddings = embeddings[word_indices]

    return used_embeddings

def visualize_word_embeddings_from_saved(model_dir, folder_path, num_words=1):
    # 获取前 num_words 个频次最高的词汇和 JSON 字符串
    sorted_token_frequencies, _ = get_word_list(folder_path, top_n=num_words)

    # 提取前 num_words 个词汇
    word_list = [token for token, freq in sorted_token_frequencies]

    # 加载保存的词嵌入和词汇表
    embeddings = np.load(os.path.join(model_dir, "word_embeddings.npy"))
    vocab = load_vocab(vocab_path)  # 获取模型的词汇表，假设是一个字典形式

    # 获取这些词汇对应的嵌入
    used_embeddings = get_embeddings_for_word_list(word_list, vocab, embeddings)

    # 对词向量进行t-SNE降维
    reduced_embeddings = tsne_embeddings(used_embeddings)

    # 可视化t-SNE结果
    plot_tsne_embeddings(reduced_embeddings, word_list, colors=define_token_colors(word_list))



def visualize_word_embeddings_after_training(trainer, num_words=936):
    # 提取模型训练后的词嵌入
    embeddings = extract_word_embeddings(trainer.model)

    # 词汇表 word_list
    word_list = load_vocab(vocab_path)  # 获取模型的词汇列表

    # 取字典前 num_words 个键
    word_list = list(word_list.keys())[:num_words]  # 仅取前 num_words 个词

    # 对词向量进行t-SNE降维
    reduced_embeddings = tsne_embeddings(embeddings[:num_words])

    # 可视化t-SNE结果
    plot_tsne_embeddings(reduced_embeddings, word_list)


if __name__ == '__main__':
    visualize_word_embeddings_from_saved(model_dir, dataset_path, num_words=1000)
