# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2024/1/30 8:54
import torch

def array2tensor(data_temp):
    data_temp = torch.from_numpy(data_temp).float()
    return data_temp