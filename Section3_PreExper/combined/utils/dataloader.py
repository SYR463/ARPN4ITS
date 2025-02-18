import os
import json
from functools import partial

from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F


class CBOWDataset(Dataset):
    """
    自定义数据集类，用于加载 CBOW 模型的训练数据。
    :param cbow_input: CBOW 模型输入数据
    :param cbow_output: CBOW 模型输出数据
    """

    def __init__(self, cbow_input, cbow_output):
        self.cbow_input = cbow_input
        self.cbow_output = cbow_output

    def __len__(self):
        return len(self.cbow_input)

    def __getitem__(self, idx):
        """
        根据索引返回 CBOW 输入和输出数据。
        """
        cbow_x = self.cbow_input[idx]
        cbow_y = self.cbow_output[idx]
        return cbow_x, cbow_y


class SkipGramDataset(Dataset):
    """
    自定义数据集类，用于加载 Skip-Gram 模型的训练数据。
    :param skipgram_input: Skip-Gram 模型输入数据
    :param skipgram_output: Skip-Gram 模型输出数据
    """

    def __init__(self, skipgram_input, skipgram_output):
        self.skipgram_input = skipgram_input
        self.skipgram_output = skipgram_output

    def __len__(self):
        return len(self.skipgram_input)

    def __getitem__(self, idx):
        """
        根据索引返回 Skip-Gram 输入和输出数据。
        """
        skipgram_x = self.skipgram_input[idx]
        skipgram_y = self.skipgram_output[idx]
        return skipgram_x, skipgram_y


def collate_cbow(batch, text_pipeline=None):
    """
    处理 CBOW 模型的批次数据
    :param batch: 输入的批次数据，格式为 [(context, target), ...]
    :param text_pipeline: 用于将文本转换为索引的函数
    :return: 返回输入和输出张量
    """
    # 分离输入和输出
    cbow_inputs, cbow_targets = zip(*[(item[0], item[1]) for item in batch])

    # 将输入和输出转换为张量
    # 对 CBOW 输入，期望它是一个上下文的列表
    cbow_inputs = [torch.tensor(context, dtype=torch.long) for context in cbow_inputs]
    cbow_targets = torch.tensor(cbow_targets, dtype=torch.long)

    # Padding 如果必要，可以对每个上下文进行填充
    max_len = max(len(context) for context in cbow_inputs)
    cbow_inputs_padded = torch.zeros(len(cbow_inputs), max_len, dtype=torch.long)
    for i, context in enumerate(cbow_inputs):
        cbow_inputs_padded[i, :len(context)] = context

    return cbow_inputs_padded, cbow_targets


def collate_skipgram(batch, text_pipeline=None):
    """
    处理 Skip-Gram 模型的批次数据
    :param batch: 输入的批次数据，格式为 [(center_word, context), ...]
    :param text_pipeline: 用于将文本转换为索引的函数
    :return: 返回输入和输出张量
    """
    # 分离输入和输出
    skipgram_inputs, skipgram_targets = zip(*[(item[0], item[1]) for item in batch])

    # 将输入转换为张量
    skipgram_inputs = torch.tensor(skipgram_inputs, dtype=torch.long)

    # 为每个输入词扩展上下文词，并展平
    skipgram_inputs_expanded = []
    skipgram_targets_flat = []

    for context, center_word in zip(skipgram_targets, skipgram_inputs):
        # 对每个中心词扩展目标
        skipgram_inputs_expanded.extend([center_word] * len(context))  # 每个中心词与多个目标一一映射
        skipgram_targets_flat.extend(context)  # 将目标（上下文词）添加到目标列表中

    # 将扩展后的输入和目标转换为1D张量
    skipgram_inputs_expanded = torch.tensor(skipgram_inputs_expanded, dtype=torch.long)
    skipgram_targets_flat = torch.tensor(skipgram_targets_flat, dtype=torch.long)

    return skipgram_inputs_expanded, skipgram_targets_flat



def get_dataloader_and_vocab(model_name, data_dir, batch_size, shuffle, vocab=None):
    """
    获取用于训练的 DataLoader 和词汇表
    :param model_name: 模型类型，'cbow' 或 'skipgram'
    :param data_dir: 存放上下文数据的目录
    :param batch_size: 每个批次的样本数量
    :param shuffle: 是否打乱数据
    :param vocab: 词汇表（如果没有提供，将从文件加载）
    :return: 返回 DataLoader 和 vocab
    """

    # 加载词汇表
    vocab = load_vocab("/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json")

    cbow_input, cbow_output, skipgram_input, skipgram_output = [], [], [], []

    # 加载单一文件
    # cbow_input, cbow_output, skipgram_input, skipgram_output = load_and_prepare_data(
    #     os.path.join(data_dir, "tokens_vehicle_data_1118846987000_RTree.txt"), vocab)

    # 批量加载文件：从文件夹中依次读取每个文件内容，并加载训练数据
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            # 加载数据并准备CBOW和Skip-Gram输入输出
            file_cbow_input, file_cbow_output, file_skipgram_input, file_skipgram_output = load_and_prepare_data(file_path, vocab)
            cbow_input.extend(file_cbow_input)
            cbow_output.extend(file_cbow_output)
            skipgram_input.extend(file_skipgram_input)
            skipgram_output.extend(file_skipgram_output)

    # 创建自定义数据集，根据模型类型选择合适的 collate_fn
    if model_name == "cbow":
        dataset = CBOWDataset(cbow_input, cbow_output)
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        dataset = SkipGramDataset(skipgram_input, skipgram_output)
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")

    # 使用 DataLoader 加载数据
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn)
    )

    return dataloader, vocab


def load_vocab(file_path):
    """从JSON文件加载词汇表"""
    with open(file_path, 'r') as file:
        vocab = json.load(file)
    return vocab


def map_to_vocab(node, vocab):
    """
    将节点映射为词汇表中的索引
    :param node: 节点的字符串表示
    :param vocab: 词汇表
    :return: 词汇表中的索引
    """
    return vocab.get(node, vocab["<UNK>"])


def prepare_data(context_lines, vocab):
    """
    根据上下文生成CBOW或Skip-Gram模型的输入输出样本。
    :param context_lines: 节点上下文数据，每行格式如：<NL> B3 D6 <L> B3 C4 <L> C5 D6
    :param vocab: 词汇表
    :return: CBOW或Skip-Gram的输入输出样本
    """
    cbow_input, cbow_output, skipgram_input, skipgram_output = [], [], [], []

    for line in context_lines:
        nodes = line.strip().split(" ")

        # 将节点映射为词汇表中的索引
        nodes = [map_to_vocab(node, vocab) for node in nodes]

        # 判断第一个token是否为<NL>，用于决定是使用Skip-Gram还是CBOW
        if nodes[0] == vocab["<NL>"]:
            # 使用Skip-Gram：用[<NL> C2 D3]预测[<L> C2 D3] 和 [<L> C2 E6]
            for i in range(0, 3):
                skipgram_input.append(nodes[i])  # 当前节点作为输入
                skipgram_output.append(nodes[:i] + nodes[i + 1:])  # 上下文作为输出
        elif nodes[0] == vocab["<L>"]:
            # 使用CBOW：用[<L> C2 D3]预测[<NL> C2 D3]
            for i in range(0, 3):
                cbow_input.append(nodes[:i] + nodes[i + 1:])  # 上下文作为输入
                cbow_output.append(nodes[i])  # 当前节点作为输出

    return cbow_input, cbow_output, skipgram_input, skipgram_output


def prepare_data_all(context_lines, vocab):
    """
    根据上下文生成CBOW或Skip-Gram模型的输入输出样本。
    :param context_lines: 节点上下文数据，每行格式如：<NL> B3 D6 <L> B3 C4 <L> C5 D6
    :param vocab: 词汇表
    :return: CBOW或Skip-Gram的输入输出样本
    """

    cbow_input, cbow_output, skipgram_input, skipgram_output = [], [], [], []

    for line in context_lines:
        nodes = line.strip().split(" ")

        # 将节点映射为词汇表中的索引
        nodes = [map_to_vocab(node, vocab) for node in nodes]

        # 构建Skip-Gram以及CBOW数据集
        for i in range(0, 3):
            skipgram_input.append(nodes[i])  # 当前节点作为输入
            skipgram_output.append(nodes[:i] + nodes[i + 1:])  # 上下文作为输出
            cbow_input.append(nodes[:i] + nodes[i + 1:])  # 上下文作为输入
            cbow_output.append(nodes[i])  # 当前节点作为输出

    return cbow_input, cbow_output, skipgram_input, skipgram_output


def load_and_prepare_data(file_path, vocab):
    """
    从文件中读取节点上下文，并生成训练数据。
    :param file_path: 上下文数据文件路径
    :param vocab: 词汇表
    :return: CBOW 和 Skip-Gram 的训练样本
    """

    # 读取上下文数据
    with open(file_path, "r") as file:
        context_lines = file.readlines()

    # 生成训练样本
    cbow_input, cbow_output, skipgram_input, skipgram_output = prepare_data(context_lines, vocab)

    return cbow_input, cbow_output, skipgram_input, skipgram_output


def save_data_to_file(data, file_path):
    """
    将数据保存到文件中
    :param data: 需要保存的数据
    :param file_path: 文件路径
    """

    # 确保目录存在，如果不存在则创建
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w") as file:
        for item in data:
            if isinstance(item, int):  # 如果 item 是整数，直接写入
                file.write(f"{item}\n")
            else:  # 如果 item 是可迭代对象（如列表），则 join
                file.write(" ".join(map(str, item)) + "\n")


def save_prepared_data(cbow_input, cbow_output, skipgram_input, skipgram_output, output_dir):
    """
    保存CBOW和Skip-Gram训练数据
    :param cbow_input: CBOW模型的输入数据
    :param cbow_output: CBOW模型的输出数据
    :param skipgram_input: Skip-Gram模型的输入数据
    :param skipgram_output: Skip-Gram模型的输出数据
    :param output_dir: 输出文件夹
    """
    save_data_to_file(cbow_input, f"{output_dir}/cbow_input.txt")
    save_data_to_file(cbow_output, f"{output_dir}/cbow_output.txt")
    save_data_to_file(skipgram_input, f"{output_dir}/skipgram_input.txt")
    save_data_to_file(skipgram_output, f"{output_dir}/skipgram_output.txt")


# if __name__ == '__main__':
#     # 设置词汇表和上下文数据文件路径
#     vocab = load_vocab('../../../vocab/vocab.json')
#     context_file_path = "../../../dataPreprocess/outputRTreeTokenContext/tokens_vehicle_data_1118846987000_RTree.txt"  # 上下文文件路径
#     output_dir = "../dataset"  # 输出文件夹
#
#     # 读取数据并准备训练数据
#     cbow_input, cbow_output, skipgram_input, skipgram_output = load_and_prepare_data(context_file_path, vocab)
#
#     # 保存准备好的数据
#     save_prepared_data(cbow_input, cbow_output, skipgram_input, skipgram_output, output_dir)
#
# if __name__ == '__main1__':
#     # 加载词汇表
#     vocab = load_vocab('../../../vocab/vocab.json')
#     node = "<L>"
#     # 映射节点到词汇表中的索引
#     index = map_to_vocab(node, vocab)
#
#     print(f"The index of node '{node}' is: {index}")


if __name__ == '__main__':
    context_lines = """
    <NL> B2 D4 <L> B2 C3 <L> C3 D4
    <L> B2 C3 <NL> B2 D4 <L> C3 D4
    <L> C3 D4 <NL> B2 D4 <L> B2 C3
    """;
    prepare_data_all(context_lines=context_lines, vocab=load_vocab("/mnt/d/project/python/ARPN4ITS/vocab/vocab_preExper.json"));
