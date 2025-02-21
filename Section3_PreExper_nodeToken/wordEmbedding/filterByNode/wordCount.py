import os
import shutil
from collections import Counter
import json

def count_tokens_in_folder(folder_path):
    """
    统计指定文件夹中所有文件中token的出现频次
    :param folder_path: 文件夹路径
    :return: token频次字典
    """
    token_counter = Counter()  # 用来存储每个token的出现频次

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):  # 确保是文件
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # 统计每一行的token
                for line in lines:
                    tokens = line.strip().split()  # 假设每个token之间用空格分隔
                    token_counter.update(tokens)

    return token_counter


def get_word_list(folder_path, top_n=None):
    """
    输入一个文件夹，依次读取文件中的文件
    计算文件中 node token 出现次数，降序排列，放入 list 中，返回 JSON 字符串
    :param folder_path: 文件夹路径
    :param top_n: 可选参数，指定返回的前n个频次最高的词汇
    :return: 排序后的 token 频次 JSON 字符串
    """
    token_frequencies = count_tokens_in_folder(folder_path)

    # 按照频次从大到小排序
    sorted_token_frequencies = token_frequencies.most_common()

    # 如果指定了 top_n，筛选前 top_n 个
    if top_n:
        sorted_token_frequencies = sorted_token_frequencies[:top_n]

    # 将结果转换为 JSON 字符串
    json_result = json.dumps(sorted_token_frequencies, ensure_ascii=False, indent=4)

    return sorted_token_frequencies, json_result


if __name__ == '__main__':
    # 使用示例
    folder_path = "/mnt/d/project/Java/rtree_construct/preExperData/RTreeToken"
    # token_frequencies = count_tokens_in_folder(folder_path)
    #
    # # 按照频次从大到小排序
    # sorted_token_frequencies = token_frequencies.most_common()
    #
    # # 打印每个token的出现频次
    # for token, freq in sorted_token_frequencies:
    #     print(f"{token}: {freq}")

    sorted_token_frequencies, _ = get_word_list(folder_path)
    print(sorted_token_frequencies)
