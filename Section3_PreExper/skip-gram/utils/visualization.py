import json
import os

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from index2vec.utils.dataloader import load_vocab


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


def plot_tsne_embeddings(reduced_embeddings, word_list, labels=None, title="t-SNE of Skip-Gram", output_path="tsne_output.png"):
    """
    绘制t-SNE降维后的词向量
    :param reduced_embeddings: t-SNE降维后的二维词向量
    :param word_list: 词汇列表
    :param labels: 词汇的标签或类别（可选）
    :param title: 图标题
    """
    plt.figure(figsize=(12, 8))

    if labels is None:
        plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
    else:
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels_encoded, palette="viridis",
                        legend="full")

    for i, word in enumerate(word_list):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=10, alpha=0.7)

    plt.title(title)

    # 设置x轴和y轴的显示范围，调整这部分以只聚焦中心区域
    plt.xlim([2.28, 2.34])  # 调整为适合的x轴范围
    plt.ylim([1.86, 1.91])  # 调整为适合的y轴范围

    # 保存图片
    plt.savefig(output_path)
    plt.close()  # 关闭图形，以释放内存





def visualize_word_embeddings_from_saved(model_dir, num_words=36):
    # 加载保存的词嵌入和词汇表
    embeddings = np.load(os.path.join(model_dir, "word_embeddings.npy"))
    word_list = load_vocab("/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json")  # 获取模型的词汇列表

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
    word_list = load_vocab("/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json")  # 获取模型的词汇列表

    # 取字典前 num_words 个键
    word_list = list(word_list.keys())[:num_words]  # 仅取前 num_words 个词

    # 对词向量进行t-SNE降维
    reduced_embeddings = tsne_embeddings(embeddings[:num_words])

    # 可视化t-SNE结果
    plot_tsne_embeddings(reduced_embeddings, word_list)


if __name__ == '__main__':
    visualize_word_embeddings_from_saved('/mnt/d/project/python/ARPN4ITS/Section3_PreExper/skip-gram/weights/skipgram_preExper', num_words=36)
