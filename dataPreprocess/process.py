"""
原CSV文件内容：
Vehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y,Global_X,Global_Y,v_length,v_Width,v_Class,v_Vel,v_Acc,Lane_ID,Preceding,Following,Space_Headway,Time_Headway,Location
2224,6548,1902,1113437421700,41.429,472.901,6042814.264,2133542.012,14.3,6.9,2,26.54,-0.76,4,2208,2211,53.34,2.01,i-80
2283,8002,1821,1113437567100,55.072,1124.713,6042739.135,2134191.387,15.9,5.9,2,6.57,3.93,5,2330,2344,46.38,7.06,i-80
2252,6364,2150,1113437403300,54.277,174.692,6042863.582,2133247.751,15.3,6.9,2,7.33,0.0,5,2117,2266,35.98,4.91,i-80

保留需要的列：
Vehicle_ID,Global_Time,Local_X,Local_Y

离散化处理：
Vehicle_ID,Global_Time,Grid_X,Grid_Y 【# 定义网格大小  x = 13.7428, y = 15】

构建MBR映射
Vehicle_ID,Global_Time,Grid_X,Grid_Y, Min_X,Min_Y,Max_X,Max_Y

根据时间戳，划分为不同的csv文件，文件命名 vehicle_data_{Global_Time}.csv
Vehicle_ID,Global_Time,Grid_X,Grid_Y, Min_X,Min_Y,Max_X,Max_Y

"""

import pandas as pd
import numpy as np
import os


# 读取 CSV 文件
def read_csv(input_file, columns=None):
    """ 读取 CSV 文件，并选择所需列 """
    df = pd.read_csv(input_file)
    if columns:
        df = df[columns]
    return df


# 写 CSV 文件
def save_csv(df, output_file):

    """ 保存 DataFrame 到 CSV 文件，若输出目录不存在则创建 """
    output_dir = os.path.dirname(output_file)
    # 如果目录不存在，则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


# 进行数据预处理
def preprocess_data(df, grid_size_x=13.7428, grid_size_y=15):

    # 修改两者的起始位置，Local_X 以及 Local_Y 从 0 开始
    min_Local_X = df['Local_X'].min()
    df['Local_X'] = df['Local_X'] - min_Local_X
    min_Local_Y = df['Local_Y'].min()
    df['Local_Y'] = df['Local_Y'] - min_Local_Y

    # 对 Local_X 和 Local_Y 进行离散化，转换为 Grid_X 和 Grid_Y，向下取整
    df['Grid_X'] = np.floor(df['Local_X'] / grid_size_x).astype(int)
    df['Grid_Y'] = np.floor(df['Local_Y'] / grid_size_y).astype(int)

    # 为每个车辆构建 MBR 矩形框，车辆的MBR即为网格的左下、右上坐标
    df['Min_X'] = df['Grid_X'].apply(num_to_alpha)
    df['Min_Y'] = df['Grid_Y']
    df['Max_X'] = (df['Grid_X'] + 1).apply(num_to_alpha)
    df['Max_Y'] = df['Grid_Y'] + 1

    # 只保留必要的列
    df = df[['Vehicle_ID', 'Global_Time', 'Grid_X', 'Grid_Y',
             'Min_X', 'Min_Y', 'Max_X', 'Max_Y']]
    return df


# 创建一个映射字典，0 -> 'A', 1 -> 'B', ..., 25 -> 'Z', 26 -> 'AA', 27 -> 'AB', ...
def num_to_alpha(num):
    result = ""
    while num >= 0:
        result = chr(num % 26 + ord('A')) + result
        num = num // 26 - 1
    return result


# 直接按 Global_Time 进行分割
def split_and_save_by_time(df, output_dir):
    """
    按 Global_Time 进行划分，每个 Global_Time 单独存储到一个 CSV 文件
    """
    unique_times = df['Global_Time'].unique()

    for time in unique_times:
        # 获取该 Global_Time 的所有数据
        time_filtered_df = df[df['Global_Time'] == time]

        # 生成文件名
        output_file = os.path.join(output_dir, f"vehicle_data_{time}.csv")

        # 保存该时间戳的数据
        save_csv(time_filtered_df, output_file)


# 主处理函数
def process_vehicle_data(input_file, output_dir):
    # 读取文件
    df = read_csv(input_file, columns=['Vehicle_ID', 'Global_Time', 'Local_X', 'Local_Y'])

    # 预处理数据
    processed_df = preprocess_data(df)

    # 按时间戳分割并保存
    split_and_save_by_time(processed_df, output_dir)


if __name__ == "__main__":
    input_file = "rawData/us101_full.csv"  # 输入 CSV 文件路径
    output_dir = "processedData"  # 处理后文件的存储目录

    process_vehicle_data(input_file, output_dir)
