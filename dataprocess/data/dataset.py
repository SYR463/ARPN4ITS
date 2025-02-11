import pandas as pd
from torch.utils.data import Dataset
import numpy as np


class VehicleDataPreprocessor:
    def __init__(self, data, grid_size=(6, 120)):
        """
        :param data: 输入的DataFrame数据，包含Vehicle_ID, Global_Time, Local_X, Local_Y
        :param grid_size: tuple, 表示(M, N)网格的大小，M为车道数，N为每个车道的最大网格数
        """
        self.data = data
        self.grid_size = grid_size

    def preprocess(self):
        """
        将数据离散化，映射到网格中
        """
        # 确保数据按时间排序
        self.data.sort_values(by='Global_Time', inplace=True)

        # 初始化用于保存网格化后的数据
        self.grid_data = []

        # 获取唯一的时间片
        time_slices = self.data['Global_Time'].unique()

        for t in time_slices:
            grid = np.zeros(self.grid_size)  # 初始化一个空的网格
            current_data = self.data[self.data['Global_Time'] == t]

            for _, row in current_data.iterrows():
                m, n = int(row['Grid_X']) - 1, int(row['Grid_Y']) - 1  # 注意索引从0开始
                grid[m, n] = row['Vehicle_ID']  # 使用Vehicle_ID来标识该位置有车辆

            self.grid_data.append({'time': t, 'grid': grid})

    def get_preprocessed_data(self):
        return self.grid_data



class VehicleDataset(Dataset):
    def __init__(self, preprocessed_data, grid_size=(6, 120), sequence_length=3):
        """
        :param preprocessed_data: 经过预处理的离散化数据，包含'time'和'grid'
        :param grid_size: tuple, 表示(M, N)网格的大小
        :param sequence_length: int, 序列长度
        """
        self.preprocessed_data = preprocessed_data
        self.grid_size = grid_size
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        self._prepare_sequences()

    def _prepare_sequences(self):
        """
        根据预处理后的数据，使用滑动窗口构建序列和目标。
        如果时间片不连续，则用0填充。
        """
        # print(self.preprocessed_data)

        # time_slices = [item['Global_Time'] for item in self.preprocessed_data]
        # 获取唯一的时间片
        time_slices = self.preprocessed_data['Global_Time'].unique()
        continuous_slices = []


        for i in range(len(time_slices) - self.sequence_length):
            # 检查时间片是否连续
            if ((time_slices[i + self.sequence_length - 1] - time_slices[i])
                    == (self.sequence_length - 1) * (time_slices[1] - time_slices[0])):
                continuous_slices.append(i)
            else:
                # 如果不连续，则跳过这个时间片或使用0填充
                continue

        for i in continuous_slices:
            sequence_slices = time_slices[i:i + self.sequence_length]
            target_slice = time_slices[i + self.sequence_length]

            # 初始化用于保存序列中每个时间片的网格分布矩阵
            sequence_data = []

            for t in sequence_slices:
                grid = np.zeros(self.grid_size)  # 初始化一个空的网格
                current_data = self.preprocessed_data[self.preprocessed_data['Global_Time'] == t]

                for _, row in current_data.iterrows():
                    m, n = int(row['Grid_X']) - 1, int(row['Grid_Y']) - 1  # 注意索引从0开始
                    grid[m, n] = row['Grid_X']

                sequence_data.append(grid)

            # 处理目标时间片
            target_grid = np.zeros(self.grid_size)
            target_data = self.preprocessed_data[self.preprocessed_data['Global_Time'] == target_slice]

            for _, row in target_data.iterrows():
                m, n = int(row['Grid_X']) - 1, int(row['Grid_Y']) - 1
                target_grid[m, n] = row['Vehicle_ID']

            self.sequences.append(np.array(sequence_data))
            self.targets.append(target_grid)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            'sequences': self.sequences[idx],
            'targets': self.targets[idx]
        }



if __name__ == '__main__':
    # 示例数据
    data = pd.DataFrame({
        'Vehicle_ID': [515, 2127, 1033, 1890, 744],
        'Global_Time': [1118848075000, 1118847624800, 1118848324700, 1118849672700, 1118848181200],
        'Local_X': [30.034, 19.632, 6.202, 53.514, 28.878],
        'Local_Y': [188.062, 1775.614, 1701.144, 817.521, 490.086],
        'Grid_X': [3, 2, 1, 5, 3],
        'Grid_Y': [12, 118, 113, 54, 32]
    })

    # 创建预处理器实例
    preprocessor = VehicleDataPreprocessor(data)
    preprocessor.preprocess()

    # 获取预处理后的数据
    processed_data = preprocessor.get_preprocessed_data()
    print(processed_data)


if __name__ == '__main__':
    # 假设 processed_data 是已经预处理好的数据
    processed_data = preprocessor.get_preprocessed_data()

    # 创建数据集实例
    dataset = VehicleDataset(processed_data)

    # 获取一个数据项
    example = dataset[0]
    print("输入序列 shape:", example['sequences'].shape)
    print("目标 shape:", example['targets'].shape)


