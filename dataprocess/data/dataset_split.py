import pandas as pd
from sklearn.model_selection import train_test_split

# 加载数据，这里假设数据已经在一个名为'data.csv'的CSV文件中
data = pd.read_csv('processed/processed_us101_0801.csv')

# 按Global_Time列排序
data.sort_values('Global_Time', inplace=True)

# 划分训练集、验证集和测试集
# 首先划分训练集和其它（验证集和测试集）
train_data, temp_data = train_test_split(data, test_size=0.3, shuffle=False)

# 然后从临时数据中划分验证集和测试集
validation_data, test_data = train_test_split(temp_data, test_size=(2/3), shuffle=False)

# 显示分割后的数据集大小
print("训练集大小:", len(train_data))
print("验证集大小:", len(validation_data))
print("测试集大小:", len(test_data))

# 保存数据集到CSV文件中
train_data.to_csv('../dataset/train_data.csv', index=False)
validation_data.to_csv('../dataset/validation_data.csv', index=False)
test_data.to_csv('../dataset/test_data.csv', index=False)

print("数据集已保存到CSV文件中。")
