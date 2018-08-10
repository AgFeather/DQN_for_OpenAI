import numpy as np
from SumTree import SumTree

class Memory(object):
    '''
    记忆SumTree和各种参数
    '''
    epsilon = 0.01 # 平滑项，避免记忆的优先级为0
    alpha = 0.6 # 用来将记忆数据的训练loss转换成优先级的系数，0~1之间
    beta = 0.4 # importance-sampling，从初始值不断增加到1
    beta_increment = 0.001 # beta每次采样增加的幅度
    beta_max = 1.0
    error_max = 1. # 用来clipping的error最大值

    def __init__(self, capacity):
        self.tree = SumTree(capacity) # 声明容量初始化记忆库，用一个SumTree表示

    def store(self, data):
        '''对于从未被训练过的数据，取当前SumTree中最大优先级进行存储，并且更新SumTree'''
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.error_max
        self.tree.add(max_p, data)

    def sample(self, batch_size):
        '''从SumTree中进行采样, 采样batch_size个数据''' 
        batch_idx = np.empty((batch_size,), dtype=np.int32) # 存取每个数据再SumTree中的index
        batch_memory = np.empty((batch_size, self.tree.data[0].size)) # 存取数据值
        ISWeights = np.empty((batch_size, 1))   # 权重
        priority_seg = self.tree.total_p / batch_size #划分采样的优先级区间
        self.beta = np.min([self.beta_max, self.beta + self.beta_increment])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        for i in range(batch_size):
            interval_min, interval_max = priority_seg * i, priority_seg * (i + 1) # 计算当前区间的两个端点
            temp_priority = np.random.uniform(interval_min, interval_max) # 从当前区间随机生成一个优先值
            idx, pri, data = self.tree.get_leaf(temp_priority)
            prob = pri / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta) # 当前数据权重
            batch_idx[i], batch_memory[i, :] = idx, data
        return batch_idx, batch_memory, ISWeights

    def batch_update(self, tree_idx, new_errors):
        '''对于一个记忆数据被训练完后，将新的数据更新到SumTree中'''
        new_errors += self.epsilon
        clipped_errors = np.minimum(new_errors, self.error_max)
        prioritis = np.power(clipped_errors, self.alpha)
        for i, p in zip(tree_idx, prioritis):
            self.tree.update(i, p)