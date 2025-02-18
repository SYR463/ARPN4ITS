import torch.nn as nn

from Section3_PreExper_nodeToken.wordEmbedding.utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM


class CBOW_Model(nn.Module):
    """
    Implementation of CBOW model described in paper:
    https://arxiv.org/abs/1301.3781
    """

    def __init__(self, vocab_size: int, skipgram_embeddings=None):
        super(CBOW_Model, self).__init__()

        # 如果提供了Skip-Gram的嵌入，则使用它初始化CBOW的嵌入层
        if skipgram_embeddings is not None:
            self.embeddings = skipgram_embeddings  # 使用Skip-Gram嵌入
        else:
            # 如果没有提供Skip-Gram嵌入，则使用随机初始化
            self.embeddings = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=EMBED_DIMENSION,
                max_norm=EMBED_MAX_NORM,
            )

        # 定义线性层
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        # 获取词嵌入并计算均值
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x





class SkipGram_Model(nn.Module):
    """
    Implementation of Skip-Gram model described in paper:
    https://arxiv.org/abs/1301.3781
    """
    def __init__(self, vocab_size: int):
        super(SkipGram_Model, self).__init__()
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