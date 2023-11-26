import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
# mpl.rcParams['font.sans-serif'] = ['Times New Roman']
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'bold',
        'size': 16}
x = range(0,1000,1)

# 修改搜索结果存储excel的文件名
data = pd.read_excel(r"..\Data\Search_process_1.xlsx",index_col=0)
y = data.iloc[0:1000,:].values

plt.xlabel("迭代轮数", fontsize=13)
plt.ylabel("目标函数值", fontsize=13)
#绘图
plt.plot(x,y)
#展示图形
plt.show()