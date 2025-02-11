'''
从原始的NGSIM中提取 ‘us101’数据，
删除重复列，以及 ["O_Zone", "D_Zone", "Int_ID", "Section_ID", "Direction", "Movement"]
删除摩托车，v_class=1
删除位于匝道的车辆，Lane_ID为6，7，8

注意：长度的单位为英尺
'''

import pandas as pd


def openfile(path):
    df = pd.read_csv(path)
    return df


def save(df, savepath):
    # 保存处理后的数据到新的CSV文件
    df.to_csv(savepath, index=False)
    print("数据处理完成，已保存到" + savepath)

def filterData(df):
    # 筛选出 Location 为 'us-101' 的记录
    df_us_101 = df[df['Location'] == 'us-101']

    # 删除重复记录
    df_us_101 = df_us_101.drop_duplicates()

    # 删除指定的字段
    columns_to_drop = ["O_Zone", "D_Zone", "Int_ID", "Section_ID", "Direction", "Movement"]
    df_us_101 = df_us_101.drop(columns=columns_to_drop)

    # 删除 v_Class=1 的字段，v_Class=1 表示摩托车
    df_us_101 = df_us_101[df_us_101['v_Class'] != 1]

    # 删除Lane_ID为6，7，8的字段，这些字段表示车辆位于匝道，不在当前研究范围内
    df_us_101 = df_us_101[df_us_101['Lane_ID'] != 6]
    df_us_101 = df_us_101[df_us_101['Lane_ID'] != 7]
    df_us_101 = df_us_101[df_us_101['Lane_ID'] != 8]

    return df_us_101


if __name__ == '__main__':
    filepath = 'Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20240701.csv'
    df = openfile(filepath)
    df = filterData(df)

    savePath = './us101_full.csv'
    save(df, savePath)

