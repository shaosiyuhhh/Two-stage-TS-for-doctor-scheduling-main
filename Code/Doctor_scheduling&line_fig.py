from matplotlib import pyplot as plt
from waiting_approx import *
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'bold',
        'size': 16}
x = range(0,168,1)
# 修改存储医生排班计划的文件名
data = pd.read_excel(r"..\Data\work_schedule_2.xlsx", index_col=0)
y1 = data.loc['Sum'].tolist()
print(y1)

y2 = time_cal(y1)
fig, ax1 = plt.subplots()
plt.xlabel("时间段", fontsize=13)
#绘图
ax1.plot(x,y1,label='医生人数')
ax1.set_ylabel("医生人数", fontsize=13)
ax1.legend(loc=2,fontsize=13)

ax2 = ax1.twinx()
ax2.set_ylabel("排队时间", fontsize=13)
ax2.plot(x,y2,label='排队时间',color = 'orange')
ax2.legend(loc=1,fontsize=13)
#展示图形
plt.show()