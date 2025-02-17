"""
https://www.cnblogs.com/chentiao/p/18350353
词汇表构建

MBR Token:
MBR 定义为 {(x_min, y_min), (x_max, y_max)}
因此，此处用(x, y)表示一个词汇
x 属于 {A, B, C, D, ……}
y 属于 {0, 1, 2, 3, ……}

Type Token:
叶节点 <L>
非叶节点 <NL>

Spectial Token:
起始标记 <BEG>
结束标记 <END>
占位标记 <S>

"""

import json

# 创建一个映射字典，0 -> 'A', 1 -> 'B', ..., 25 -> 'Z', 26 -> 'AA', 27 -> 'AB', ...
def num_to_alpha(num):
    result = ""
    while num >= 0:
        result = chr(num % 26 + ord('A')) + result
        num = num // 26 - 1
    return result


def create_MBR_vocab(vocab, M, N):
    """
    创建 MBR 词汇表
    :param vocab: 空词汇表
    :return: MBR词汇表
    """
    word = ""
    for i in range(N):
        for j in range(M):
            word = num_to_alpha(i) + str(j)
            vocab.append(word)
    return vocab


if __name__ == '__main__':

    # 预定义的网格个数
    M = 6
    N = 5

    vocab_filename = 'vocab_preExper.json'

    vocab = []
    vocab = create_MBR_vocab(vocab, M, N)
    # 添加其他特殊标记（节点类型标记 以及 特殊标记）
    vocab = ["<UNK>", "<L>", "<NL>", "<BEG>", "<END>", "<S>"] + vocab

    # 词到索引映射
    word_to_index = {}

    # 为特殊标记分配索引 1 到 6
    special_tokens = ["<UNK>", "<L>", "<NL>", "<BEG>", "<END>", "<S>"]
    for idx, token in enumerate(special_tokens, start=0):
        word_to_index[token] = idx

    # # 为剩余的词汇分配索引，从17开始
    # for idx, word in enumerate(vocab[6:], start=17):  # 忽略前6个特殊标记
    #     word_to_index[word] = idx

    for idx, word in enumerate(vocab[6:], start=6):  # 忽略前6个特殊标记
        word_to_index[word] = idx

    # 保存词汇表
    with open(vocab_filename, 'w') as f:
        json.dump(word_to_index, f)

    # 加载词汇表
    with open(vocab_filename, 'r') as f:
        word_to_index = json.load(f)

    print(word_to_index)

