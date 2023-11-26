import random
import gurobipy as gp
from gurobipy import GRB,Model,quicksum,tupledict,tuplelist
from waiting_approx import *

# 建立医院排班系统模型
class AssignModel:
    def __init__(self, doctor_num, weekday, schedule, alpha=1.5, iter=400,
                  max_tabu_num=8, max_remain_gap = 800):
        self.d_num = doctor_num  # 医生数量
        self.K = list(range(doctor_num))
        self.A = list(range(weekday))
        for i in range(len(self.A)):
            self.A[i] += 1
        self.N = 76
        self.T = list(range(weekday * 24))   # 7*24=168
        self.S = list(range(weekday * schedule))  # 7*76=532
        self.R = np.zeros(shape=[len(self.S), len(self.T)])  # 532*168
        self.schedule_num = schedule
        self.alpha = alpha
        self.best_obj = 0
        self.tabu_list = []
        self.solution = {}  # 当前解,字典格式存储，解结构见下文所示
        self.opt_solution = {}  # 当前最优解,字典格式存储
        self.opt_fitness = ''  # 当前最优解适应度,字典格式存储
        self.iteration_history = pd.DataFrame(columns=['solution', 'fitness', '是否可行', '当前最好解'])  # 存储每次迭代结果，pandasDataframe存储
        self.neighbor_solutions = []  # 当前解的领域解，列表存储
        self.neighbor_fitness = []  # 当前解的领域解目标列表，列表存储
        self.neighbor_moves = []  # 当前解的领域解移动路线
        self.neighbor = pd.DataFrame(columns=['solution', 'fitness', 'move'])  # 当前解的领域解，pandasDataframe存储
        self.it = iter  # 迭代次数
        self.max_tabu_num = max_tabu_num  # 禁忌表中最大禁忌次数
        self.max_remain_gap = max_remain_gap  # 最大保持不变的代数
        self.get_schedule_matrix()

    def get_schedule_matrix(self):
        self.R = np.zeros(shape=[len(self.S), len(self.T)])
        for i in range(7):
            self.R[0][i]=1
        for start in np.arange(7,17):
            for i in np.arange(6*start - 41,6*start - 35):
                for k in range((i-1)%6+3):
                    self.R[i][start+k]=1
        for i in np.arange(61, 66):
            for k in range((i - 1) % 6 + 3):
                self.R[i][17+k] = 1
        for i in np.arange(66, 70):
            for k in range((i - 1) % 6 + 3):
                self.R[i][18+k] = 1
        for i in np.arange(70, 73):
            for k in range((i - 1) % 6 + 3):
                self.R[i][19+k] = 1
        for i in np.arange(73, 75):
            for k in range((i - 1) % 6 + 3):
                self.R[i][20+k] = 1
        for k in range(3):
            self.R[75][21+k] = 1
        r = self.R.copy()
        for t in np.arange(1, 7, 1):
            for i in range(76):
                for k in range(24):
                    self.R[t * 76 + i][t * 24 + k] = r[i][k]
        return self.R

    def get_schedule(self,sol):
        schedule = np.zeros(shape=[len(self.K), len(self.T)])
        for i in self.K:
            for s in self.S:
                if sol[i][s] == 1:
                    for t in self.T:
                        if self.R[s][t] == 1:
                            schedule[i][t] = 1
        return schedule

    def cal_obj(self,sol):
        p_t = [0 for _ in self.T]
        for t in self.T:
            for i in self.K:
                for n in self.S:
                    p_t[t] += sol[i][n] * self.R[n][t]
        L_t = time_cal(p_t)
        obj = sum(L_t) + self.alpha * sum(p_t)
        return obj

    def check_individual_daily_f(self,sol,i,m):
        flag = True
        # 一天最多两个白班
        job = 0
        for n in np.arange(self.N*m - self.N+1,self.N*m):
            job += sol[i][n]
        if job >= 3:
            flag = False
        # 夜班结束休息24小时
        if flag and m!=7:
            d = m - 1
            if sol[i][d*self.N] == 1:
                for n in np.arange(self.N*d+1,self.N*(d+1)):
                    if sol[i][n] == 1:
                        flag = False
                        break
        # 两班间隔不少于2小时
        if flag:
            d = m-1
            for n in np.arange(76*d+1, 76*d+76):
                for t in np.arange(24 * d + 7, 24 * d + 22):
                    if sol[i][n]*self.R[n][t] > 0:
                        for h in np.arange(76*d+1, 76*d+76):
                            if n!=h and sol[i][h]*self.R[h][t+2] >= 1:
                                flag = False
                                break
        # 夜班开始前 8 小时不上班
        if flag:
            d = m-1
            job = sol[i][d * self.N]
            if job > 0 and m > 1:
                for n in np.arange(self.N * d - self.N + 1, self.N * d):
                    for t in np.arange(24 * d - 8, 24 * d):
                        job += sol[i][n] * self.R[n][t]
                if job >= 2:
                    flag = False
        # 一周最多2个夜班
        if flag:
            job = 0
            for m in range(7):
                job += sol[i][m * self.N]
            if job >= 3:
                flag = False
        # 医生一周最多工作6天
        if flag:
            work_d = 0
            for m in self.A:
                work = 0
                for t in np.arange(24*m - 24,24*m):
                    for n in np.arange(m*self.N-self.N,m*self.N):
                        work += sol[i][n]*self.R[n][t]
                if work > 0:
                    work_d += 1
            if work_d > 6:
                flag = False
        return flag

    def get_ban_list(self):
        self.ban_list = [[] for _ in self.S]
        for i in np.arange(1,self.N):
            i_list=[]
            for t in range(24):
                if self.R[i][t] == 1:
                    i_list.append(t)
            i_list_p = i_list.copy()
            i_list.append(i_list_p[-1] + 1)
            i_list.append(i_list_p[-1] + 2)
            i_list.append(i_list_p[0] - 1)
            i_list.append(i_list_p[0] - 2)
            for j in np.arange(1,self.N):
                for t in i_list:
                    if self.R[j][t] == 1 and t>=0 and t<=23 and j not in self.ban_list[i]:
                        self.ban_list[i].append(j)
            self.ban_list[i].append(0)
            if i >= 55:
                self.ban_list[i].append(76)
        for i in np.arange(0, self.N):
            self.ban_list[0].append(i)
        for k in np.arange(1, 7):
            for j in np.arange(76*k+1,76*k+76):
                self.ban_list[76*k].append(j)
            for j in np.arange(76*k-21,76*k+1):
                self.ban_list[76*k].append(j)
        for k in np.arange(1, 7):
            for i in np.arange(76*k+1, 76*k+self.N):
                for num in self.ban_list[i%76]:
                    self.ban_list[i].append(num + 76*k)
        for i in np.arange(76 * 6 + 1, 76 * 6 + self.N):
            if 532 in self.ban_list[i]:
                self.ban_list[i].remove(532)

    def get_ban_d(self,sol,doctor):
        self.ban = [[] for _ in self.K]
        for s in self.S:
            if sol[doctor][s] == 1:
                for i in self.ban_list[s]:
                    self.ban[doctor].append(i)

    def get_initial_solution(self):
        self.ini_sol = np.zeros(shape=[self.d_num,len(self.S)])
        t_list = [2,60]
        for i in range(2):
            self.ini_sol[i][0] = 1
            for k in np.arange(2,7):
                for j in t_list:
                    self.ini_sol[i][j + k*76] = 1
        for i in np.arange(2,4):
            self.ini_sol[i][76] = 1
            for k in np.arange(3,7):
                for j in t_list:
                    self.ini_sol[i][j + k*76] = 1
        for i in np.arange(4, 6):
            self.ini_sol[i][76*2] = 1
            for j in t_list:
                self.ini_sol[i][j] = 1
            for k in np.arange(4,7):
                self.ini_sol[i][27 + k*76] = 1
        for i in np.arange(6, 8):
            self.ini_sol[i][76*3] = 1
            for k in np.arange(0,2):
                for j in t_list:
                    self.ini_sol[i][j + k*76] = 1
            for k in np.arange(5,7):
                self.ini_sol[i][27 + k*76] = 1
        for i in np.arange(8, 10):
            self.ini_sol[i][76*4] = 1
            for k in np.arange(1,3):
                for j in t_list:
                    self.ini_sol[i][j + k*76] = 1
        for i in np.arange(10, 12):
            self.ini_sol[i][76*5] = 1
            for k in np.arange(0,4):
                self.ini_sol[i][27 + k*76] = 1
        for i in np.arange(12, 14):
            self.ini_sol[i][76*6] = 1
            for k in np.arange(0,5):
                self.ini_sol[i][27 + k*76] = 1
        self.cur_obj = self.cal_obj(self.ini_sol)
        best_sol = self.ini_sol
        self.best_obj = self.cur_obj
        self.get_ban_list()
        self.get_ban_d(best_sol, 14)
        print('初始解：',self.best_obj)
        return best_sol

    def build_model(self):
        "创建模型"
        model = gp.Model("1")
        x_list = []
        for i in self.K:
            for n in self.S:
                x_list.append((i, n))
        x = {(i, n): model.addVar(lb=0, vtype=GRB.BINARY, name='x' + str([i, n])) for (i, n) in
             x_list}
        z_list = []
        for i in self.K:
            for m in range(7):
                z_list.append((i, m))
        z_list = tuplelist(z_list)
        z = {(i, m): model.addVar(lb=0, vtype=GRB.BINARY, name='z' + str([i, m])) for (i, m) in
             z_list}
        e = {t: model.addVar(lb=0, vtype=GRB.INTEGER, name='y' + str(t)) for t in self.T}
        e_ = {t: model.addVar(lb=0, vtype=GRB.INTEGER, name='y' + str(t)) for t in self.T}

        obj_expr = 0
        for t in self.T:
            obj_expr += e[t]
            obj_expr += e_[t]
        model.setObjective(obj_expr, GRB.MINIMIZE)
        # 一天最多2个白班
        for m in range(7):
            for i in self.K:
                model.addConstr(quicksum(x[i,n] for n in np.arange(self.N * m + 1, self.N * m + self.N)) <= 2)
        # 夜班结束休息24h
        for m in range(6):
            for i in self.K:
                model.addConstr(x[i,m * self.N] + quicksum(x[i,n] for n in np.arange(self.N * m + 1, self.N * (m + 1) + 1)) <= 2)
        model.addConstrs(
            x[i, 6 * self.N] + quicksum(x[i, n] for n in np.arange(self.N * 6 + 1, self.N * 7)) <= 2 for i in self.K)
        # 两班间隔不少于2小时
        for i in self.K:
            for d in range(7):
                for n in np.arange(76 * d, 76 * d + 76):
                    for t in np.arange(24 * d + 7, 24 * d + 22):
                        for h in np.arange(76 * d, 76 * d + 76):
                            if h == n:
                                continue
                            model.addConstr(x[i,n] * self.R[n][t] + x[i,h] * self.R[h][t + 2] <= 1)

        # 夜班开始前8小时不上班
        for i in self.K:
            for m in range(1,7):
                model.addConstrs(x[i,m * self.N] + quicksum(x[i, n] * self.R[n][t] for n in np.arange(self.N * m - self.N + 1, self.N * m)) <= 1  for t in np.arange(24 * m - 8, 24 * m))

        # 夜班次数 <= 2
        for i in self.K:
            model.addConstr(quicksum(x[i,m * self.N] for m in range(7)) <= 2)

        # 每个时间段工作人数 + 工作人数 >=1
        for t in self.T:
            model.addConstr(quicksum(x[i, n] * self.R[n][t] for i in self.K for n in self.S) == self.pt[t] + e[t]- e_[t])
            model.addConstr(quicksum(x[i, n] * self.R[n][t] for i in self.K for n in self.S) >= 1)

        # 医生一周最多工作6天
        M = 9
        for i in self.K:
            for m in range(7):
                model.addConstr(quicksum(x[i,n] * self.R[n][t] for t in np.arange(24 * m, 24 * m + 24) for n in np.arange(m * self.N, m * self.N + self.N)) >= z[i, m])

                model.addConstr(quicksum(x[i,n] * self.R[n][t] for t in np.arange(24 * m, 24 * m + 24) for n in
                                         np.arange(m * self.N, m * self.N + self.N)) <= M * z[i, m])
        for i in self.K:
            model.addConstr(quicksum(z[i, m] for m in range(7)) <= 6)
        model.optimize()
        self.x_sol = np.zeros((len(self.K), len(self.S)))
        for i in self.K:
            for n in self.S:
                self.x_sol[i, n] = model.getVarByName("x" + str([i, n])).X

    def get_pt(self, pt_):
        self.pt = pt_


random.seed(1)
hospital = AssignModel(doctor_num=15, weekday=7, schedule=76, alpha=1.5, max_tabu_num=8, max_remain_gap = 800)
sol = hospital.get_initial_solution()
sche = hospital.get_schedule(sol)
sche = pd.DataFrame(sche)
sche.loc['Sum'] = sche.apply(lambda x: x.sum())
data = sche
pt_0 = data.loc['Sum'].to_list()
wt_0 = time_cal(pt_0)
init_obj = sum(pt_0)*1.5 + sum(wt_0)
b_obj = init_obj

iter = 0
wt = wt_0
pt = pt_0
obj_best = 1e5

while iter < 1000:
    ratio = [0 for _ in range(168)]
    up_list = []
    down_list = []
    pt_ = pt.copy()
    for i in range(168):
        ratio[i] = wt[i] / 1.5 / pt[i]
        if ratio[i] > 1:
            up_list.append(i)
        else:
            down_list.append(i)
    while True:
        start_t = random.randint(0, 165)
        length_t = random.randint(3, 8)
        if start_t + length_t <= 168:
            break
    up = 0
    down = 0
    for i in range(length_t):
        if start_t + i in up_list:
            up += 1
        else:
            down += 1
    if up > down:
        for i in range(length_t):
            pt_[start_t + i] += 1
        wt_ = time_cal(pt_)
        cur_obj = sum(pt_) * 1.5 + sum(wt_)
        if cur_obj < b_obj:
            pt = pt_.copy()
            b_obj = cur_obj
    else:
        flag = True
        for i in range(length_t):
            if pt_[start_t + i] > 1:
                pt_[start_t + i] -= 1
            else:
                flag = False
        if flag:
            wt_ = time_cal(pt_)
            cur_obj = sum(pt_) * 1.5 + sum(wt_)
            if cur_obj < b_obj:
                pt = pt_.copy()
                b_obj = cur_obj
    if iter % 50 == 49:
        hospital.get_pt(pt)
        hospital.build_model()
        obj = hospital.cal_obj(hospital.x_sol)
        if obj < obj_best:
            x_best = hospital.x_sol
            obj_best = obj
        print('轮次：', iter, '第二阶段搜索解：', obj)
    iter += 1
    print('轮次：', iter, '搜索解：', b_obj)

pd.DataFrame(x_best).to_excel(r'..\Data\initial_result2.xlsx')
