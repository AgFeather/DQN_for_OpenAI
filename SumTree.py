import numpy as np

class SumTree(object):
    '''
    定义DQN中的Memory存储结构 SumTree
    这里使用一个np.array表示整棵树
    '''
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1) # 整棵树的所有节点，前c-1个节点为父节点，后c个节点为子节点，树中的叶子节点用来保存优先级
        self.data = np.zeros(capacity, dtype=object) # 存储记忆数据的叶子节点，大小等于记忆库大小

    def add(self, p, data):
        '''
        将新的记忆添加到tree中
        :param p: 添加数据的优先级
        :param data: 要被添加的数据
        '''
        tree_idx = self.data_pointer + self.capacity - 1 # 对应该数据的优先级叶子节点在树中的位置。
        self.data[self.data_pointer] = data # 将数据放入数据存储array中
        self.update(tree_idx, p) # 将该叶子节点的优先级更新到树种

        self.data_pointer += 1
        if self.data_pointer >= self.capacity: #数据的指针大小在内存大小范围内循环。
            # self.data_pointer = 0
            # print('Memory is full.. begin to cover old memory')
            self.data_pointer %= self.capacity

    def update(self, tree_idx, p):
        '''
        当一个记忆数据被用来训练时，会产生新的优先值p，根据新的p值更新树
        :param tree_idx: 传入数据优先级在树种对应的位置
        :param p: 数据的优先级
        '''
        change = p - self.tree[tree_idx] # 根据原有的优先级和更新的优先级计算改变的大小
        self.tree[tree_idx] = p
        while tree_idx != 0:  # 从叶子节点循环向上更新父节点存储的优先级，知道根节点为止
            tree_idx = (tree_idx - 1) // 2 # 父节点的index = (index-1)//2
            self.tree[tree_idx] += change

    def get_leaf(self, priority):
        '''
        根据选取的优先值进行抽样
        :param priority: 本次抽样的目标优先级
        :return: 数据在tree中的index，该数据的优先级，以及该数据本身
        '''
        parent_idx = 0 # 记录父节点的index，根据SumTree规则，从根节点开始
        while True:
            left_idx = 2 * parent_idx + 1 # 找到当前节点对应的左子树
            right_idx = left_idx + 1 # 找到右子树
            if left_idx >= len(self.tree): # 左子树已经是叶子节点
                leaf_idx = parent_idx
                break
            else:
                if priority <= self.tree[left_idx]: # 如果左节点的值大于p，递归进左节点
                    parent_idx = left_idx
                elif priority <= self.tree[right_idx]: # 如果左节点的值小于p，右节点的值大于p，进右节点
                    parent_idx = right_idx
                else:
                    priority -= self.tree[right_idx] # 两个节点都小于p，减去右节点的值，递归进左节点
                    parent_idx = left_idx

        data_idx = leaf_idx - self.capacity + 1 # 计算对应的data的index
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        '''
        获取全部优先级的和（根节点的值）
        '''
        return self.tree[0]