'''
从数据集中提取如下信息，并将数据映射到一个唯一的网格中
Grid_X = Lane_ID  每个网格宽68.714/5
Grid_Y = Local_Y/15  每个网格长 15feet，总长2235.252/15 = 149.01 + 1

Processed Data:
   Vehicle_ID    Global_Time  Local_X   Local_Y  Grid_X  Grid_Y
0         515  1118848075000   30.034   188.062       3      12
1        2127  1118847624800   19.632  1775.614       2     118
2        1033  1118848324700    6.202  1701.144       1     113
3        1890  1118849672700   53.514   817.521       5      54
4         744  1118848181200   28.878   490.086       3      32

'''

import pandas as pd
import numpy as np

# 假设数据存储在 'data.csv' 文件中
file_path = 'raw/us101_full.csv'

# 加载数据
data = pd.read_csv(file_path)

# 查看数据集的前几行，确保数据加载正确
print("Original Data:")
print(data.head())

# 网格参数
grid_size = 15  # 每个网格的宽度为15英尺

# 计算网格位置
# data['Grid_X'] = data['Lane_ID'].astype(int)
# data['Grid_Y'] = (data['Local_Y'] // grid_size).astype(int)

# 保留所需的列
filtered_data = data[['Vehicle_ID', 'Global_Time', 'v_Vel', 'v_Acc', 'Local_X', 'Local_Y', 'Lane_ID']]

# 查看处理后的数据
print("\nProcessed Data:")
print(filtered_data.head())

# 保存处理后的数据到新的CSV文件
output_file_path = 'processed/processed_us101.csv'
filtered_data.to_csv(output_file_path, index=False)

print(f"\nProcessed data saved to {output_file_path}")
