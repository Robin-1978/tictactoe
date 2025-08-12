import numpy as np
from game import Game  # 补充导入

class TreeNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # key: move (row, col) 元组, value: TreeNode
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0  # 先验概率
    
    def get_value(self, c_puct):
        # 计算UCB值
        if self.parent is None:
            u = 0
        else:
            u = c_puct * self.prior * np.sqrt(self.parent.visits) / (1 + self.visits)
        return (self.value_sum / self.visits) + u if self.visits > 0 else u
    
    def select_child(self, c_puct):
        # 选择UCB最大的子节点
        return max(self.children.items(), key=lambda item: item[1].get_value(c_puct))
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def expand(self, action_probs):
        # 根据动作概率扩展子节点
        for action, prob in action_probs:
            # 确保action是元组类型
            if isinstance(action, list):
                action = tuple(action)
            if action not in self.children:
                self.children[action] = TreeNode(self.state, self)
                self.children[action].prior = prob

class MCTS:
    def __init__(self, policy_value_net, c_puct=1.5, n_playout=100):
        self.root = TreeNode(None)
        self.policy_value_net = policy_value_net
        self.c_puct = c_puct  # 探索系数
        self.n_playout = n_playout  # 每次决策的模拟次数
        self.device = policy_value_net.device
        # 获取棋盘大小
        self.board_size = policy_value_net.board_size
        
    def _playout(self, game):
        node = self.root
        states = []
        moves = []
        
        # 选择阶段：从根节点到叶节点
        while not node.is_leaf():
            move, node = node.select_child(self.c_puct)
            moves.append(move)
            game.make_move(move)
            states.append(game.get_state())
        
        # 检查游戏是否结束（先判断胜负，再判断是否满）
        end = False
        winner = None
        if moves:  # 确保moves非空（有落子才可能获胜）
            if game.check_win(*moves[-1]):
                end = True
                winner = game.current_player  # 获胜方是上一个落子的玩家（因为current_player已切换）
        if not end and game.is_full():
            end = True
            winner = 0  # 平局
        
        if end:
            # 终端节点价值计算（当前玩家视角：若对手获胜，价值为-1）
            if winner == 0:
                value = 0.0
            else:
                value = -1.0 if winner == game.current_player else 1.0
            # 回溯更新
            for _ in moves:
                node = node.parent
                node.visits += 1
                node.value_sum += value
                value = -value  # 切换视角
            return
        
        # 扩展阶段：用模型预测策略和价值
        policy, value = self.policy_value_net.predict(np.array([game.get_state()]))
        policy = np.exp(policy[0])  # 从log_softmax转换为概率
        legal_moves = game.get_legal_moves()
        # 只保留合法落子的概率，非法的设为0
        action_probs = []
        for move in legal_moves:
            idx = move[0] * self.board_size + move[1]  # 使用动态棋盘大小
            action_probs.append((move, policy[idx]))
        # 扩展子节点
        node.expand(action_probs)
        
        # 回溯更新
        value = value[0][0]  # 模型预测的价值（当前玩家视角）
        for _ in moves:
            node = node.parent
            node.visits += 1
            node.value_sum += value
            value = -value  # 切换视角
    
    def get_move_probs(self, game, temp=1e-3):
        # 多次模拟后返回落子概率
        for _ in range(self.n_playout):
            # 创建Game副本时传递棋盘大小和获胜条件
            game_copy = Game(
                board_size=self.board_size,
                win_length=getattr(game, 'win_length', 3)
            )
            game_copy.board = np.copy(game.board)
            game_copy.current_player = game.current_player
            self._playout(game_copy)
        
        # 从根节点子节点的访问次数计算概率
        move_visits = [(move, node.visits) for move, node in self.root.children.items()]
        moves, visits = zip(*move_visits) if move_visits else ([], [])
        
        # 极端情况处理：如果没有有效落子
        if len(visits) == 0:
            return [], []
        
        # 转换为数组并确保数值有效性
        # visits = np.array(visits, dtype=np.float64)
        visits = np.array(visits, dtype=np.float64) + 1e-8  # 加微小值避免0

        
        # 强制确保至少有一个有效访问计数（核心修复）
        if np.all(visits == 0):
            visits = np.ones_like(visits)  # 全部设为1，避免全零
        else:
            visits = np.maximum(visits, 1e-10)  # 非零情况下确保最小值
        
        # 确保温度参数有效
        temp = max(temp, 1e-3)  # 温度不小于0.001
        
        # 计算概率（带完整异常处理）
        try:
            # 使用对数和指数进行计算，提高数值稳定性
            log_visits = np.log(visits)
            log_probs = log_visits / temp
            # 在计算exp之前先进行归一化，减去最大值防止溢出
            log_probs -= np.max(log_probs)  # 减去最大值，避免数值溢出
            probs = np.exp(log_probs)
            probs /= np.sum(probs)  # 归一化概率，确保总和为1
            
            # 计算总和并检查有效性
            total = probs.sum()
            if total <= 1e-10 or not np.isfinite(total):
                # 总和无效时使用均匀分布
                probs = np.ones_like(probs) / len(probs)
            else:
                # 正常归一化
                probs /= total
            
            # 最后检查并替换可能的NaN/inf值
            probs = np.nan_to_num(probs, nan=1.0/len(probs), posinf=1.0, neginf=0.0)
            # 再次归一化确保总和为1
            probs /= probs.sum()
            
        except Exception as e:
            # 任何异常都使用均匀分布作为保底
            probs = np.ones_like(visits) / len(visits)

        # print(f"Legal moves: {moves}, Probs: {probs}")
        return moves, probs
    
    def update_with_move(self, move):
        # 移动根节点到下一个状态
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            # 若落子不在子节点中（如对手落子），重新初始化根节点
            self.root = TreeNode(None)