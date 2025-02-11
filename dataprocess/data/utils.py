"""
数据处理，工具类
"""

import os
import pandas as pd

def read_file(file_path):
    """
    读取文件内容并返回
    """
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def display_file_content(file_path):
    """
    显示文件内容
    """
    content = read_file(file_path)
    print(content)

def list_files_in_directory(directory_path):
    """
    列出目录中的所有文件
    """
    if os.path.exists(directory_path):
        files = os.listdir(directory_path)
        for file in files:
            print(file)
    else:
        print(f"目录 {directory_path} 不存在")

def read_csv_as_dataframe(file_path):
    """
    读取CSV文件并返回DataFrame
    """

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        print(f"文件 {file_path} 不存在")
        return None

def display_csv_head(file_path, n=5):
    """
    显示CSV文件的前n行内容
    """
    df = read_csv_as_dataframe(file_path)
    if df is not None:
        print(df.head(n))

# 示例用法
if __name__ == "__main__":
    directory = 'splitByGlobalTime'
    list_files_in_directory(directory)
    file_path = os.path.join(directory, 'data_1118848075000.csv')
    display_file_content(file_path)
    display_csv_head(file_path, 5)
