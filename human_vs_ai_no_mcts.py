from game import Game
from model import PolicyValueNet
import numpy as np
import torch


class AIPlayerNoMCTS:
    def __init__(self, model_path, player_id):
        self.policy_value_net = PolicyValueNet()
        # 加载检查点，提取模型状态字典
        checkpoint = torch.load(model_path, map_location=self.policy_value_net.device)
        # 检查是否包含嵌套的模型状态字典
        if 'model_state_dict' in checkpoint:
            self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.policy_value_net.load_state_dict(checkpoint)
        self.policy_value_net.eval()
        self.player_id = player_id  # 记录AI的玩家编号（1或2）
        
    def get_move(self, game):
        # 直接使用模型预测进行决策，不使用MCTS
        state = game.get_state()
        policy, _ = self.policy_value_net.predict(np.array([state]))
        policy = np.exp(policy[0])  # 从log_softmax转换为概率
        
        # 获取所有合法落子
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None
        
        # 计算每个合法落子的概率
        move_probs = []
        for move in legal_moves:
            idx = move[0] * 3 + move[1]  # 3x3棋盘的索引计算
            move_probs.append((move, policy[idx]))
        
        # 选择概率最高的落子
        move_probs.sort(key=lambda x: x[1], reverse=True)
        return move_probs[0][0]


def human_vs_ai_no_mcts():
    game = Game()
    # 加载模型
    try:
        model_path = "checkpoint/temp_model_3400.pth"
    except FileNotFoundError:
        print("Error: 未找到模型文件，请先运行trainer.py训练模型")
        return
    
    # 选择先手方（谁执黑子）
    while True:
        choice = input("请选择先手方（黑子）：1-玩家先行，2-AI先行 (输入1或2，q退出)：")
        if choice.lower() == 'q':
            print("游戏已退出")
            return
        if choice in ["1", "2"]:
            choice = int(choice)
            break
        print("输入无效，请重新选择！")
    
    # 根据选择设置玩家和AI的编号
    if choice == 1:
        human_player = 1  # 玩家执黑子（X）
        ai_player = 2     # AI执白子（O）
        current_player = 1  # 玩家先行
        print("你执黑子（X）先行，AI执白子（O）后行")
    else:
        human_player = 2  # 玩家执白子（O）
        ai_player = 1     # AI执黑子（X）
        current_player = 1  # AI先行（因为AI是1号，黑子）
        print("AI执黑子（X）先行，你执白子（O）后行")
    
    # 初始化AI玩家（传入其编号）
    ai = AIPlayerNoMCTS(model_path, ai_player)
    
    while True:
        game.render()
        if current_player == human_player:
            # 人类落子
            legal_moves = game.get_legal_moves()
            print("请输入落子坐标（格式：行 列），或输入q退出游戏")
            # print(f"合法落子: {[tuple(m) for m in legal_moves]}")
            
            try:
                user_input = input("你的选择: ").strip().lower()
                # 检查是否退出
                if user_input == 'q':
                    print("游戏已退出")
                    return
                # 直接使用行列坐标
                row, col = map(int, user_input.split())
                move = (row, col)
                if move not in [tuple(m) for m in legal_moves]:
                    raise ValueError("无效坐标")
            except (ValueError):
                print("输入格式错误，请重试")
                continue    
            # 执行人类落子
            game.make_move(move)

            # 检查人类是否获胜
            if game.check_win(*move):
                game.render()
                print("恭喜，你赢了！")
                break
            # 切换到AI
            current_player = ai_player
        else:
            # AI落子（不使用MCTS）
            move = ai.get_move(game)
            if not move:
                break  # 棋盘满了
            print(f"AI落子: {move}")
            game.make_move(move)
            # 检查AI是否获胜
            if game.check_win(*move):
                game.render()
                print("AI赢了，再接再厉！")
                break
            # 切换到人类
            current_player = human_player
        
        # 检查平局
        if game.is_full():
            game.render()
            print("平局！")
            break

if __name__ == "__main__":
    human_vs_ai_no_mcts()