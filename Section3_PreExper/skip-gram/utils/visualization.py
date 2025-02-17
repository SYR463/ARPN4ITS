import json
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from index2vec.utils.dataloader import load_vocab

vocab_path = "/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json"
tsne_output_path = "tsne_output.png"
model_dir = "/mnt/d/project/python/ARPN4ITS/Section3_PreExper/skip-gram/weights/skipgram_preExper"
pic_title = "t-SNE of Skip-Gram"

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

def plot_tsne_embeddings(reduced_embeddings, word_list, title=pic_title, output_path=tsne_output_path):
    """
    绘制t-SNE降维后的词向量
    :param reduced_embeddings: t-SNE降维后的二维词向量
    :param word_list: 词汇列表
    :param title: 图标题
    """
    plt.figure(figsize=(12, 8))

    # 为每个词汇分配颜色
    colors = []
    deep_brown_tokens = {"A3", "E5", "D4", "B4"}  # 深棕色token集合
    dark_green_tokens = {"B0", "E3", "C2", "D3", "B1", "C2", "D0", "E1"}  # 墨绿色token集合

    for word in word_list:
        if word in deep_brown_tokens:
            colors.append((197/255, 90/255, 17/255))  # 深棕色
        elif word in dark_green_tokens:
            colors.append((0/255, 188/255, 18/255))  # 墨绿色
        else:
            colors.append((180/255, 199/255, 231/255))  # 浅蓝色

    # 绘制散点图
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=colors, s=50, alpha=0.7)

    # 添加词汇标签
    for i, word in enumerate(word_list):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=10, alpha=0.7)

    plt.title(title)

    # # 设置x轴和y轴的显示范围，调整这部分以只聚焦中心区域
    # plt.xlim([-1.5, -1.15])  # 调整为适合的x轴范围
    # plt.ylim([-2.1, -1.6])  # 调整为适合的y轴范围

    plt.savefig(output_path)
    plt.close()  # 关闭图形，以释放内存






def visualize_word_embeddings_from_saved(model_dir, num_words=36):
    # 加载保存的词嵌入和词汇表
    embeddings = np.load(os.path.join(model_dir, "word_embeddings.npy"))
    word_list = load_vocab(vocab_path)  # 获取模型的词汇列表

    # 取字典前 num_words 个键
    word_list = list(word_list.keys())[:num_words]  # 仅取前 num_words 个词

    # 对词向量进行t-SNE降维
    reduced_embeddings = tsne_embeddings(embeddings[:num_words])

    # 可视化t-SNE结果
    plot_tsne_embeddings(reduced_embeddings, word_list)



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
    visualize_word_embeddings_from_saved(model_dir, num_words=36)
