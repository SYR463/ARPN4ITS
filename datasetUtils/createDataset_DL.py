import os
import random
import shutil


def process_content(content):
    """
    处理树结构内容，按顺序将每行的节点拼接成一个字符串，并在末尾添加 <END> 标记。
    :param content: 文件的原始内容
    :return: 处理后的内容
    """
    # 将每一行按换行符分开
    lines = content.strip().split("\n")

    # 去掉每行的首尾空格并将所有节点拼接成一个列表
    result = [line.strip() for line in lines]

    # 在最后添加 <END> 标记
    result.append('<END>')

    # 返回连接后的字符串
    return ' '.join(result)


if __name__ == '__main1__':
    content = """
    <NL>C2D3
      <L>C2D3
    <NL>F2D3
      <L>F2D3
      <L>F1D3
    """

    res = process_content(content)
    print(res)


def create_file_if_not_exists(file_path):
    """
    如果文件不存在，则创建该文件。
    :param file_path: 需要创建的文件路径
    """
    if not os.path.exists(file_path):
        # 确保文件所在的目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # 创建空文件
        open(file_path, 'w').close()


def split_dataset(src_folder, train_file, valid_file, test_file,
                  train_ratio=0.6, valid_ratio=0.2):
    """
    将R*-tree文件夹中的文件划分为训练集、验证集和测试集，并预留API接口处理每个文件内容。
    :param src_folder: 包含R*-tree文件的源文件夹
    :param train_file: 训练集保存的txt文件路径
    :param valid_file: 验证集保存的txt文件路径
    :param test_file: 测试集保存的txt文件路径
    :param train_ratio: 训练集比例
    :param valid_ratio: 验证集比例
    """

    # 获取文件夹中所有文件的列表
    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    # # 随机打乱文件顺序
    # random.shuffle(all_files)

    # 根据比例计算划分的文件数
    total_files = len(all_files)
    train_size = int(train_ratio * total_files)
    valid_size = int(valid_ratio * total_files)

    # 划分训练集、验证集和测试集
    train_files = all_files[:train_size]
    valid_files = all_files[train_size:train_size + valid_size]
    test_files = all_files[train_size + valid_size:]

    # 创建输出文件及其目录
    create_file_if_not_exists(train_file)
    create_file_if_not_exists(valid_file)
    create_file_if_not_exists(test_file)

    # 将文件内容写入对应的txt文件
    def write_to_file(files, output_file):
        with open(output_file, 'w') as outf:
            for file in files:
                file_path = os.path.join(src_folder, file)
                with open(file_path, 'r') as f:
                    content = f.read().strip()  # 读取并去除两端的空白字符
                    processed_content = process_content(content)  # 调用处理接口
                    outf.write(processed_content + '\n')  # 每个文件内容写成一行

    # 写入训练集、验证集和测试集文件
    write_to_file(train_files, train_file)
    write_to_file(valid_files, valid_file)
    write_to_file(test_files, test_file)


if __name__ == '__main__':

    # 该部分内容为深度学习的数据集，即学习R-tree之间的关系，因此此处的输入为03_RTreeToken，而不是其上下文序列04_RTreeTokenContext
    src_folder = '/mnt/d/project/Java/rtree_construct/us101Data/03_RTreeToken'
    train_file = '/mnt/d/project/Java/rtree_construct/us101Data/05_RTreeTokenDataset/train.txt'  # 训练集保存路径
    valid_file = '/mnt/d/project/Java/rtree_construct/us101Data/05_RTreeTokenDataset/valid.txt'  # 验证集保存路径
    test_file = '/mnt/d/project/Java/rtree_construct/us101Data/05_RTreeTokenDataset/test.txt'  # 测试集保存路径

    # src_folder = '/mnt/d/project/python/ARPN4ITS/dataPreprocess/test1'  # 你的R*-tree文件夹路径
    # train_file = '/mnt/d/project/python/ARPN4ITS/dataPreprocess/testSplit/train.txt'  # 训练集保存路径
    # valid_file = '/mnt/d/project/python/ARPN4ITS/dataPreprocess/testSplit/valid.txt'  # 验证集保存路径
    # test_file = '/mnt/d/project/python/ARPN4ITS/dataPreprocess/testSplit/test.txt'  # 测试集保存路径

    split_dataset(src_folder, train_file, valid_file, test_file)
