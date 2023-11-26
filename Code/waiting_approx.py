import numpy as np
import math
import pandas as pd

# 患者到达率文件
lam = pd.read_excel(r"..\Data\data_2.xlsx")
lam = lam.drop(['病人每小时到达率λ（人/h）'], axis=1)
lam = lam.values  # .values方法将dataframe转为numpy.ndarray，也可以用np.array(X)将其转为numpy.ndarray
lam.tolist()
lamb = [item for list in lam for item in list]

qm = 50   # 最大排队人数
miu = 6   # 服务率
time_range = 168
schedule = [2 for _ in range(7)]
for _ in range(17):
    schedule.append(8)
day_plan = schedule.copy()
for _ in range(6):
    for i in day_plan:
        schedule.append(i)

def time_cal(pt):
    A = np.zeros(shape=[time_range+1,qm + 1])
    A[0][0] = 1
    wt = [0 for _ in range(time_range)]
    for t in range(time_range):
        beta = lamb[t] + pt[t] * miu
        Nm = int(2 * beta)   # 状态转移次数
        # 转移 n 次的概率
        B = np.zeros(shape=[Nm+1, 1])
        B[0] = math.exp(-beta * 1)

        n = 0
        # 状态转移矩阵
        P = np.zeros(shape=[qm + 1, qm + 1])
        P[0][0] = pt[t] * miu / beta
        P[0][1] = lamb[t] / beta
        P[1][0] = miu / beta
        P[1][1] = (pt[t] - 1) * miu / beta
        P[1][2] = lamb[t] / beta
        P[qm][qm - 1] = min(pt[t], qm) * miu / beta
        P[qm][qm] = 1 - min(pt[t], qm) * miu / beta
        for i in np.arange(2, qm):
            P[i][i - 1] = min(pt[t], i) * miu / beta
            P[i][i] = (pt[t] * miu - min(pt[t], i) * miu) / beta
            P[i][i + 1] = lamb[t] / beta

        P_matrix_l=[0 for _ in range(Nm+1)]
        P_n = P
        P_matrix_l[0] = np.identity(qm + 1)
        for i in np.arange(1,Nm+1):
            P_matrix_l[i] = P_n
            P_n = np.dot(P_n,P)

        while n <= Nm:
            obj = 0
            A_k = A[t].copy()
            prob_i = A_k.copy()

            for k in np.arange(1,n+1):
                prob_i += np.dot(A_k,P_matrix_l[k])
            for i in range(qm+1):
                obj += max(i-pt[t],0)/(n+1) * prob_i[i]
                # 需要等待的人数 * 状态持续的时间 * 病人在各状态出现的概率
            wt[t] += obj * B[n]
            if t < time_range:
                A[t+1] += B[n]*np.dot(A_k,P_matrix_l[n])
            if n == Nm:
                break
            B[n+1] = B[n] * beta/(n+1)
            n = n + 1
        wt[t] = float(wt[t])

    # 返回各个时段等待时间，等待时间之和
    print('患者等待时间：',sum(wt))
    return wt

# s = time.time()
# wait = time_cal(schedule)
# e = time.time()
# print(wait)
# print(sum(wait))
# print(e-s)