'''
将数据集按照时间戳进行拆分, 1000ms = 1s, 采样时间 0.1s
'''

import os
import pandas as pd

# 创建存储分割数据集的文件夹
output_folder = '../dataset/splitByGlobalTime'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取CSV文件
df = pd.read_csv('processed/processed_us101.csv')

# 获取唯一的时间戳
unique_times = df['Global_Time'].unique()

# 根据时间戳分割数据集，并存储到文件夹中
for time in unique_times:
    # 筛选出当前时间戳的数据
    df_time = df[df['Global_Time'] == time]

    # 生成文件名
    file_name = os.path.join(output_folder, f'data_{time}.csv')

    # 存储分割后的数据集
    df_time.to_csv(file_name, index=False)

print(f"数据处理完成，已根据时间戳分割并保存到 '{output_folder}' 文件夹中")
