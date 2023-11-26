import random
from waiting_approx import *

# 建立医院排班系统模型
class AssignModel:
    def __init__(self, doctor_num, weekday, schedule, alpha=1.5, iter = 400, max_tabu_num=8, max_remain_gap = 800):
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
        self.tabu_table = pd.DataFrame(columns=['move', 'tabu_num'])  # 禁忌表,索引为移动,
        self.best_sol = []

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

    def cal_obj(self,sol):
        p_t = [0 for _ in self.T]
        for t in self.T:
            for i in self.K:
                for n in self.S:
                    p_t[t] += sol[i][n] * self.R[n][t]
        L_t = time_cal(p_t)
        obj = sum(L_t) + self.alpha * sum(p_t)
        return obj

    def delete_work(self, sol, i, n):
        nei_sol = sol.copy()
        nei_sol[i][n] = 0
        flag = True
        m = math.floor(n / 76)
        for t in np.arange(24 * m, 24 * m + 24):
            if self.R[n][t] == 1:
                p_t = 0
                for h in np.arange(self.N * m, self.N * m + self.N):
                    for j in self.K:
                        p_t += nei_sol[j][h] * self.R[h][t]
                if p_t == 0:
                    flag = False
        return flag, nei_sol

    def insert_night(self, sol, i, n):
        nei_sol = sol.copy()
        nei_sol[i][n] = 1
        flag = True
        work_d = 0
        for m in self.A:  # 每一天
            for n in np.arange(m * self.N - self.N, m * self.N):  # 对应第m天的班次
                if nei_sol[i][n] > 0:
                    work_d += 1
                    break
        if work_d > 6:
            flag = False

        if flag:
            job = 0
            for m in range(7):
                job += nei_sol[i][m * self.N]
            if job >= 3:
                flag = False

        return flag, nei_sol

    def insert_day(self, sol, i, n):
        nei_sol = sol.copy()
        nei_sol[i][n] = 1
        flag = True
        m = math.floor(n/76) + 1
        job = 0
        for k in np.arange(self.N * m - self.N + 1, self.N * m):
            if nei_sol[i][k] > 0:
                job += 1
            if job >= 3:
                flag = False
                break
        if flag:
            work_d = 0
            for m in self.A:  # 每一天
                for n in np.arange(m * self.N - self.N, m * self.N):  # 对应第m天的班次
                    if nei_sol[i][n] > 0:
                        work_d += 1
                        break
            if work_d > 6:
                flag = False
        return flag, nei_sol

    def get_ban_d(self,sol):
        self.ban = [[] for _ in self.K]
        for doctor in range(15):
            for s in self.S:
                if sol[doctor][s] == 1:
                    for i in self.ban_list[s]:
                        self.ban[doctor].append(i)

    def local_search(self):
        search_fig = []
        Tem = 0.02 * self.best_obj
        search_obj = self.best_obj
        search_sol = self.best_sol.copy()
        for i in range(self.it):
            cur_sol = search_sol.copy()
            doctor = random.randint(0,14)
            cur_obj = 5000
            insert_flag = False
            delete_flag = False
            operator = random.randint(0,1)  # 0:插入 1：删除
            if operator == 0:
                feas_list = []
                count = 0
                for o in self.S:
                    if o not in self.ban[doctor] and cur_sol[doctor][o] == 0:
                        feas_list.append(o)
                while True:
                    s = random.choice(feas_list)
                    count += 1
                    if s % 76 == 0:
                        flag, c_sol = self.insert_night(cur_sol, doctor, s)
                        if flag:
                            c_obj = self.cal_obj(c_sol)
                            if c_obj < cur_obj:
                                cur_obj = c_obj
                                cur_sol = c_sol
                                insert_flag = True
                                delete_flag = False
                                insert_p = s
                            break
                    else:
                        flag, c_sol = self.insert_day(cur_sol, doctor, s)
                        if flag:
                            c_obj = self.cal_obj(c_sol)
                            if c_obj < cur_obj:
                                cur_obj = c_obj
                                cur_sol = c_sol
                                insert_flag = True
                                delete_flag = False
                                insert_p = s
                            break
                    if count > 10:
                        operator = 1
                        break
            elif operator == 1:
                feas_list = []
                count = 0
                while True:
                    for o in self.S:
                        if cur_sol[doctor][o] == 1:
                            feas_list.append(o)
                    if feas_list != []:
                        break
                    else:
                        doctor = random.randint(0, 14)
                while True:
                    s = random.choice(feas_list)
                    count += 1
                    flag, c_sol = self.delete_work(cur_sol, doctor, s)
                    if flag:
                        c_obj = self.cal_obj(c_sol)
                        if c_obj < cur_obj:
                            cur_obj = c_obj
                            cur_sol = c_sol
                            insert_flag = False
                            delete_flag = True
                            delete_p = s
                        break
                    if count > 10:
                        break
            if cur_obj < search_obj:
                search_obj = cur_obj
                search_sol = cur_sol
                if cur_obj < self.best_obj:
                    self.best_obj = cur_obj
                    self.best_sol = cur_sol
                if insert_flag:
                    for id in self.ban_list[insert_p]:
                        self.ban[doctor].append(id)
                if delete_flag:
                    for id in self.ban_list[delete_p]:
                        self.ban[doctor].remove(id)

            elif cur_obj >= search_obj and cur_obj != 5000:
                if random.random() < math.exp((- cur_obj + search_obj)/Tem):
                    search_obj = cur_obj
                    search_sol = cur_sol
                    if insert_flag:
                        for id in self.ban_list[insert_p]:
                            self.ban[doctor].append(id)
                    if delete_flag:
                        for id in self.ban_list[delete_p]:
                            self.ban[doctor].remove(id)
            if i % 5 == 4:
                Tem = Tem / 1.5
            if i % 1000 == 999:
                Tem = 0.02 * self.best_obj
            print('轮次：', i, '搜索解：', cur_obj, '当前解：', search_obj, '迭代最优解：', self.best_obj)
            search_fig.append(search_obj)
        return search_fig

random.seed(0)
hospital = AssignModel(doctor_num=15, weekday=7, schedule=76, alpha=1.5, iter=4000, max_tabu_num=8, max_remain_gap = 800)
R_matrix = hospital.get_schedule_matrix()
hospital.get_ban_list()

sol = pd.read_excel(r'..\Data\initial_result2.xlsx')
sol = sol.drop(sol.columns[[0]], axis=1)
hospital.best_sol = sol.values
hospital.best_obj = hospital.cal_obj(hospital.best_sol)
hospital.get_ban_d(hospital.best_sol)

fig_data = hospital.local_search()
obj = int(hospital.best_obj)
pd.DataFrame(fig_data).to_excel('Search_process_2.xlsx')
pd.DataFrame(hospital.best_sol).to_excel('SA_search_2.xlsx')