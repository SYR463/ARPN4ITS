import torch.nn as nn

from index2vec.utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM


class CBOW_Only_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(CBOW_Only_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x


class Combined_Model(nn.Module):
    """
    Update the CBOW-Only OR Skip-Gram-Only model
    Implementation of Combined Skip-Gram With CBOW model
    实现CBOW模型，基于Skip-Gram的嵌入输入来预测非叶节点的token。
    """
    def __init__(self, vocab_size: int, skipgram_embeddings: nn.Embedding):
        super(Combined_Model, self).__init__()
        # 使用Skip-Gram训练的嵌入矩阵
        self.embeddings = skipgram_embeddings
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)  # 计算上下文的平均
        x = self.linear(x)
        return x


class SkipGram_Only_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Only_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = self.linear(x)
        return x