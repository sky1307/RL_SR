from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pulp import LpProblem, LpStatus, lpSum, LpVariable, GLPK, LpMinimize
import time
OBJ_EPSILON = 1e-12


class Game(object):
    def __init__(self, config, env, random_seed=1000):
        self.random_state = np.random.RandomState(seed=random_seed)

        self.data_dir = env.data_dir
        self.DG = env.topology.DG
        self.traffic_file = env.traffic_file
        self.traffic_matrices = env.traffic_matrices
        self.traffic_matrices_dims = self.traffic_matrices.shape
        self.tm_cnt = env.tm_cnt
        self.num_pairs = env.num_pairs
        self.pair_idx_to_sd = env.pair_idx_to_sd
        self.pair_sd_to_idx = env.pair_sd_to_idx
        self.num_nodes = env.num_nodes
        self.num_links = env.num_links
        self.link_idx_to_sd = env.link_idx_to_sd
        self.link_sd_to_idx = env.link_sd_to_idx
        self.link_capacities = env.link_capacities
        self.link_weights = env.link_weights
        self.shortest_paths_node = env.shortest_paths_node  # paths with node info
        self.shortest_paths_link = env.shortest_paths_link  # paths with link info

        self.model_type = config.model_type

        # for LP
        self.lp_pairs = [p for p in range(self.num_pairs)]
        self.lp_nodes = [n for n in range(self.num_nodes)]
        self.links = [e for e in range(self.num_links)]
        self.lp_links = [e for e in self.link_sd_to_idx]
        self.pair_links = [(pr, e[0], e[1]) for pr in self.lp_pairs for e in self.lp_links]
        self.timeout = config.timeout
        self.load_multiplier = {}


        # ~~~~~~~~~~~~~~
        self.solution = np.zeros((self.num_nodes, self.num_nodes)) -1
        self.link_utilization = np.zeros(self.num_links)
        self.link_load        = np.zeros(self.num_links)
        # self.max_moves = int(self.num_pairs * (config.max_moves / 100.))
        self.topK = []
        self.old_mlu = 0

    def get_state(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]
        self.link_load = np.zeros(self.num_links)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j :
                    continue
                if self.solution[i][j] == -1:
                    self.increaseFlow(i, j, tm[i][j])
                else:
                    e = self.solution[i][j]
                    self.increaseFlow(i, e, tm[i][j])
                    self.increaseFlow(e, j, tm[i][j])
        self.link_utilization = self.link_load / self.link_capacities
        self.old_mlu = max(self.link_utilization)
        self.topK = self.get_critical_topK_flows(tm_idx)
        self.topK100 = [ int(i*100) for i in self.topK]

        return np.array(list(self.link_utilization) + self.topK).reshape(self.num_links + self.max_moves, 1)


    # def get_state1(self, tm_idx):
    #     tm = self.traffic_matrices[tm_idx]
    #     self.link_load = np.zeros(self.num_links)
    #     for i in range(self.num_nodes):
    #         for j in range(self.num_nodes):
    #             if i == j :
    #                 continue
    #             if self.solution[i][j] == -1:
    #                 self.increaseFlow(i, j, tm[i][j])
    #             else:
    #                 e = self.solution[i][j]
    #                 self.increaseFlow(i, e, tm[i][j])
    #                 self.increaseFlow(e, j, tm[i][j])
    #     self.link_utilization = self.link_load / self.link_capacities
    #     self.old_mlu = max(self.link_utilization)
    #     self.topK = self.get_critical_topK_flows(tm_idx)
    #     self.topK = list(np.random.choice(range(self.num_pairs), len(self.topK)))
    #     return np.array(list(self.link_utilization) + self.topK).reshape(self.num_links + self.max_moves, 1)

    def increaseFlow(self, u, v, flow, link_load = None):
        if link_load is None:
            link_load = self.link_load
        if u == v :
            return 0
        pair  = self.pair_sd_to_idx[(u,v)]
        increment = flow / len(self.shortest_paths_link[pair])

        for path in self.shortest_paths_link[pair]:
            for edge in path:
                link_load[edge] += increment

    def chooseActionfromPolicy(self, policy):
        p  = policy.reshape(len(self.topK), -1)
        A = []

        alpha = 0 # % choose max action
        if np.random.random() < alpha:
            for k in range(len(self.topK)):
                A.append(np.argmax(p[k]))
        else:
            for k in range(len(self.topK)):
                A.append(np.random.choice(self.num_nodes, 1 , p = p[k]/sum(p[k]))[0])
        return A


    def DoAction(self, action, tm_idx):
        A = []
        for i in range(len(self.topK)):
            e = self.topK[i]
            u, v = self.pair_idx_to_sd[e]
            if u == action[i]:
                A.append(-1)
            elif v == action[i]:
                A.append(-2)
            else:
                A.append(action[i])
        tm = self.traffic_matrices[tm_idx]
        for i in range(len(self.topK)):
            e = self.topK[i]
            u, v = self.pair_idx_to_sd[e]
            if self.solution[u][v] == -1:
                self.increaseFlow(u,v, -tm[u][v])
            else:
                s = self.solution[u][v]
                self.increaseFlow(u, s, -tm[u][v])
                self.increaseFlow(s, v, -tm[u][v])
            a = action[i]
            if a == u or a == v:
                a = -1
            self.solution[u][v] = a
            if a == -1:
                self.increaseFlow(u, v, tm[u][v])
            else:
                self.increaseFlow(u, a, tm[u][v])
                self.increaseFlow(a, v, tm[u][v])
        self.link_utilization = self.link_load / self.link_capacities
        return A

    def get_Utilization(self):
        return self.link_utilization

    def reward(self, tm_idx):
        # no_sr = self.normal_routing(tm_idx = tm_idx)
        new_mlu = max(self.link_utilization)
        # r = 10 * (self.old_mlu/new_mlu -1)  + 0.5 / new_mlu
        r = 1 / new_mlu
        return r

    def normal_routing(self, tm_idx):
        tm = self.traffic_matrices[tm_idx]
        load_links = np.zeros(self.num_links)
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j:
                    continue
                self.increaseFlow(i, j, tm[i][j], load_links)
        return max(load_links/ self.link_capacities)

    def heuristic(self, timeLimit=500, tm_idx = 1):

        startTime = time.time() * 1000000000
        stopTime = startTime + timeLimit * 1000000

        ul = self.link_load.copy()
        s = self.solution.copy()
        tm = self.traffic_matrices[tm_idx]
        current_u = self.link_load.copy()
        current_s = self.solution.copy()
        max_old = self.old_mlu


        while (time.time() * 1000000000 < stopTime):
            flag = 0
            for pair in self.topK:
                u, v = self.pair_idx_to_sd[pair]
                for i in range(self.num_nodes):
                    a = i
                    if current_s[u][v] != -1:
                        p = current_s[u][v]
                        self.increaseFlow(u,p, -tm[u][v], current_u)
                        self.increaseFlow(p,v, -tm[u][v], current_u)
                    else:
                        self.increaseFlow(u,v, -tm[u][v], current_u)

                    self.increaseFlow(u, a, tm[u][v], current_u)
                    self.increaseFlow(a, v, tm[u][v], current_u)
                    max_new = max(current_u/self.link_capacities)
                    # print(max_new)
                    if a==v or a==u:
                        a = -1

                    current_s[u][v] = a
                    if max_new < max_old:
                        flag = 1
                        max_old = max_new
                        ul = current_u.copy()
                        s =  current_s.copy()
                    else:
                        current_u = ul.copy()
                        current_s = s.copy()
            if flag == 0:
                break

        # print(self.old_mlu,"    " ,max(ul))
        action = []
        for pair in self.topK:
            u, v = self.pair_idx_to_sd[pair]
            if s[u][v] ==-1:
                action.append(u)
            else:
                action.append(int(s[u][v]))
        # print(action)
        return action






    def generate_inputs(self, normalization=True):
        self.normalized_traffic_matrices = np.zeros(
            (self.valid_tm_cnt, self.traffic_matrices_dims[1], self.traffic_matrices_dims[2], self.tm_history),
            dtype=np.float32)  # tm state  [Valid_tms, Node, Node, History]
        idx_offset = self.tm_history - 1
        for tm_idx in self.tm_indexes:
            for h in range(self.tm_history):
                if normalization:
                    tm_max_element = np.max(self.traffic_matrices[tm_idx - h])
                    if tm_max_element > 0:
                        self.normalized_traffic_matrices[tm_idx - idx_offset, :, :, h] = self.traffic_matrices[
                                                                                             tm_idx - h] / tm_max_element  # [Valid_tms, Node, Node, History]
                    else:
                        self.normalized_traffic_matrices[tm_idx - idx_offset, :, :, h] = self.traffic_matrices[
                            tm_idx - h]  # [Valid_tms, Node, Node, History]

                else:
                    self.normalized_traffic_matrices[tm_idx - idx_offset, :, :, h] = self.traffic_matrices[
                        tm_idx - h]  # [Valid_tms, Node, Node, History]

    def get_topK_flows(self, tm_idx, pairs):
        tm = self.traffic_matrices[tm_idx]
        f = {}
        for p in pairs:
            s, d = self.pair_idx_to_sd[p]
            f[p] = tm[s][d]

        sorted_f = sorted(f.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

        cf = []
        m = min(self.max_moves, len(sorted_f))
        for i in range(m):
            cf.append(sorted_f[i][0])
        for j in range(m, self.max_moves):
            cf.append(-1)

        return cf



    def get_critical_topK_flows(self, tm_idx, critical_links=5):
        link_loads = self.link_load.copy()
        critical_link_indexes = np.argsort(-(link_loads / self.link_capacities))[:critical_links]

        cf_potential = []
        for pair_idx in range(self.num_pairs):
            u, v = self.pair_idx_to_sd[pair_idx]
            if self.solution[u][v] == -1:
                for path in self.shortest_paths_link[pair_idx]:
                    if len(set(path).intersection(critical_link_indexes)) > 0:
                        cf_potential.append(pair_idx)
                        break
            else:
                e = self.solution[u][v]
                d1 = self.pair_sd_to_idx[(u, e)]
                d2 = self.pair_sd_to_idx[(e, v)]
                for path in self.shortest_paths_link[d1]:
                    if len(set(path).intersection(critical_link_indexes)) > 0:
                        cf_potential.append(pair_idx)
                        break
                for path in self.shortest_paths_link[d2]:
                    if len(set(path).intersection(critical_link_indexes)) > 0:
                        cf_potential.append(pair_idx)
                        break

        # print(cf_potential)
        # assert len(cf_potential) >= self.max_moves, \
        #     ("cf_potential(%d) < max_move(%d), please increase critical_links(%d)" % (
        #         cf_potential, self.max_moves, critical_links))

        return self.get_topK_flows(tm_idx, cf_potential)



class CFRRL_Game(Game):
    def __init__(self, config, env, random_seed=1000):
        super(CFRRL_Game, self).__init__(config, env, random_seed)

        self.project_name = config.project_name
        self.max_moves = int(self.num_pairs * (config.max_moves / 100.))
        self.action_dim = self.max_moves * env.num_nodes
        assert self.max_moves <= self.action_dim, (self.max_moves, self.action_dim)

        self.tm_history = 1
        self.tm_indexes = np.arange(self.tm_history - 1, self.tm_cnt)
        self.valid_tm_cnt = len(self.tm_indexes)

        if config.method == 'pure_policy':
            self.baseline = {}

        self.generate_inputs(normalization=True)
        self.state_dims = [self.num_links + self.max_moves, 1]



    def advantage(self, tm_idx, reward):
        if tm_idx not in self.baseline:
            return reward
        total_v, cnt = self.baseline[tm_idx]
        return reward - (total_v / cnt)

    def update_baseline(self, tm_idx, reward):
        if tm_idx in self.baseline:
            total_v, cnt = self.baseline[tm_idx]

            total_v += reward
            cnt += 1

            self.baseline[tm_idx] = (total_v, cnt)
        else:
            self.baseline[tm_idx] = (reward, 1)
