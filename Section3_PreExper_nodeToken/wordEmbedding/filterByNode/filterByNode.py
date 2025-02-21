import os
import shutil
from collections import Counter

import os
import shutil

def copy_files_based_on_tokens(folder_path, target_folder, target_tokens):
    """
    根据指定节点的token筛选文件并将其复制到新的文件夹中
    :param folder_path: 源文件夹路径
    :param target_folder: 目标文件夹路径
    :param target_tokens: 需要筛选的目标token列表
    """
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):  # 确保是文件
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # 检查文件中的每一行是否包含所有目标token
                file_contains_target_tokens = True  # 假设文件包含所有目标token
                for token in target_tokens:
                    # 检查是否有任何一行包含这个目标token
                    token_found_in_file = False
                    for line in lines:
                        tokens = line.strip().split()
                        if token in tokens:
                            token_found_in_file = True
                            break
                    # 如果某个 token 没有找到，说明文件不符合要求
                    if not token_found_in_file:
                        file_contains_target_tokens = False
                        break

                # 如果文件包含所有目标 token，则复制该文件到目标文件夹
                if file_contains_target_tokens:
                    shutil.copy(file_path, os.path.join(target_folder, filename))
                    print(f"文件 {filename} 被复制到 {target_folder}")




if __name__ == '__main__':

    # <NL>A0E4: 1303; <NL>A1E5: 1139
    target_tokens = ["<NL>A2D5"]  # 替换为您希望筛选的目标token
    source_folder = "/mnt/d/project/Java/rtree_construct/preExperData/RTreeToken"
    # 目标文件夹路径
    destination_folder = "/mnt/d/project/Java/rtree_construct/preExperData/RTreeTokenFiltered/A2D5"

    copy_files_based_on_tokens(source_folder, destination_folder, target_tokens)
