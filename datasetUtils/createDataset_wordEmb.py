import os
import random
import shutil


def create_file_if_not_exists(file_path):
    """
    如果文件不存在，则创建该文件。
    :param file_path: 需要创建的文件路径
    """
    if not os.path.exists(file_path):
        # 确保文件所在的目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)


def split_data(source_folder, train_folder, val_folder, split_ratio=0.7):
    """
    将数据从 source_folder 中划分为训练集和验证集，并将文件复制到相应的文件夹中。
    :param source_folder: 原始数据所在文件夹路径
    :param train_folder: 训练集文件夹路径
    :param val_folder: 验证集文件夹路径
    :param split_ratio: 训练集比例，默认 70%
    """
    # 获取 source_folder 中的所有文件
    files = os.listdir(source_folder)

    # 过滤掉非文件的项，只留下文件
    files = [f for f in files if os.path.isfile(os.path.join(source_folder, f))]

    # 打乱文件顺序
    random.shuffle(files)

    # 计算训练集的文件数
    split_index = int(len(files) * split_ratio)

    # 划分训练集和验证集
    train_files = files[:split_index]
    val_files = files[split_index:]

    # 将训练集文件复制到训练集文件夹
    for file_name in train_files:
        src = os.path.join(source_folder, file_name)
        dest = os.path.join(train_folder, file_name)

        create_file_if_not_exists(dest)
        shutil.copy(src, dest)
        print(f"复制文件 {file_name} 到训练集")

    # 将验证集文件复制到验证集文件夹
    for file_name in val_files:
        src = os.path.join(source_folder, file_name)
        dest = os.path.join(val_folder, file_name)

        create_file_if_not_exists(dest)
        shutil.copy(src, dest)


if __name__ == '__main__':

    src_folder = '/mnt/d/project/Java/rtree_construct/ExperData/04_RTreeTokenContext'  # 你的R*-tree文件夹路径
    train_file = '/mnt/d/project/Java/rtree_construct/ExperData/05_RTreeTokenContext/train'  # 训练集保存路径
    valid_file = '/mnt/d/project/Java/rtree_construct/ExperData/05_RTreeTokenContext/val'  # 验证集保存路径

    split_data(src_folder, train_file, valid_file)
