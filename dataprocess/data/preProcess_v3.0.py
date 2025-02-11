'''

从数据集中提取如下信息，并将数据映射到一个唯一的网格中
Grid_X = Lane_ID  每个网格宽68.714/5
Grid_Y = Local_Y/15  每个网格长 15feet，总长2235.252/15 = 149.01 + 1

Processed Data:
   'Vehicle_ID', 'Global_Time', 'v_Vel', 'v_Acc', 'Local_X', 'Local_Y', 'Lane_ID'

按照时间戳的顺序将 Processed Data 映射到唯一网格中

将已经处理完成的 csv文件转化为 numpy 格式的 npz 文件

'''

import pandas as pd
import numpy as np

# 假设数据存储在一个 DataFrame 中
data = pd.read_csv("raw/us101_full.csv")

# 保留所需的列
data = data[['Vehicle_ID', 'Global_Time', 'v_Vel', 'v_Acc', 'Local_X', 'Local_Y', 'Lane_ID']]

# 时间戳归一化处理，归一化为 [0, 1]
# max_time = data['Global_Time'].max()
min_time = data['Global_Time'].min()
data['Global_Time'] = data['Global_Time'] - min_time

min_Local_X = data['Local_X'].min()
data['Local_X'] = data['Local_X'] - min_Local_X
min_Local_Y = data['Local_Y'].min()
data['Local_Y'] = data['Local_Y'] - min_Local_Y

# print(data)
print("车辆数据初始处理完成")

# 定义网格大小  x = 13.7428, y = 15
delta_x, delta_y = 13.8, 15

# 1. 根据时间片分割数据
time_intervals = sorted(data['Global_Time'].unique())
time_intervals = time_intervals[::2]  # 降低采样频率为 0.2s
processed_data = []

# 创建一个字典用于快速查找每个时间片的数据
time_data_dict = {t: data[data['Global_Time'] == t] for t in time_intervals}
print("时间戳字典构建完成")


for idx, t in enumerate(time_intervals):
    # 选取当前时间片的数据
    current_data = time_data_dict[t]

    # 初始化 M x N x k 矩阵，其中 k 是特征数
    M, N = 5, 150  # 具体根据区域大小确定
    k = 13  # 特征数
    grid_matrix = np.full((M, N, k), 0)  # 使用 0 进行初始化，表示该网格中没有车辆

    # 2. 对每辆车映射到网格中
    for _, row in current_data.iterrows():
        m = int(row['Local_X'] // delta_x)
        n = int(row['Local_Y'] // delta_y)

        # 填充当前时间片特征
        grid_matrix[m, n, 0] = row['Vehicle_ID']
        grid_matrix[m, n, 1] = row['Lane_ID']
        grid_matrix[m, n, 2] = row['v_Vel']
        grid_matrix[m, n, 3] = row['v_Acc']
        grid_matrix[m, n, 4] = row['Global_Time']
        grid_matrix[m, n, 5] = row['Local_X']
        grid_matrix[m, n, 6] = row['Local_Y']

        # 3. 查找前后时间片特征
        # 前一时间片
        if idx > 0:  # 如果有前一时间片
            prev_data = time_data_dict[time_intervals[idx - 1]]
            prev_row = prev_data[prev_data['Vehicle_ID'] == row['Vehicle_ID']]
            if not prev_row.empty:
                grid_matrix[m, n, 7] = prev_row['Global_Time'].values[0]
                grid_matrix[m, n, 8] = prev_row['Local_X'].values[0]
                grid_matrix[m, n, 9] = prev_row['Local_Y'].values[0]

        # 后一时间片
        if idx < len(time_intervals) - 1:  # 如果有后一时间片
            next_data = time_data_dict[time_intervals[idx + 1]]
            next_row = next_data[next_data['Vehicle_ID'] == row['Vehicle_ID']]
            if not next_row.empty:
                grid_matrix[m, n, 10] = next_row['Global_Time'].values[0]
                grid_matrix[m, n, 11] = next_row['Local_X'].values[0]
                grid_matrix[m, n, 12] = next_row['Local_Y'].values[0]

    if idx % 100 == 0:
        # 打印处理结束标识
        tempTime = t + min_time
        remaining = len(time_intervals) - idx - 1
        print(f"数据处理完成，时间戳：{tempTime}，还剩 {remaining} 个时间戳")

    # 将处理后的矩阵加入到列表中
    processed_data.append(grid_matrix)

# 最终的 processed_data 列表包含了每个时间片的 M x N x k 数据
# 保存每个时间片的矩阵到单个 .npz 文件中
np.savez('processed/us101_full_preprocessed.npz', processed_data)
