import torch
from game import Game
from model import PolicyValueNet
from mcts import MCTS
import numpy as np


class AIPlayer:
    def __init__(self, model_path, player_id):
        # 加载检查点，提取模型状态字典和参数
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # 提取网络配置参数 - 适配新架构
        board_size = checkpoint.get('board_size', 3)
        input_channels = checkpoint.get('input_channels', 64)
        res_channels = checkpoint.get('res_channels', [32, 64])
        dropout_rate = checkpoint.get('dropout_rate', 0.0)
        
        # 使用提取的参数初始化策略价值网络
        self.policy_value_net = PolicyValueNet(
            board_size=board_size,
            input_channels=input_channels,
            res_channels=res_channels,
            dropout_rate=dropout_rate
        )
        
        # 加载模型状态字典
        if 'model_state_dict' in checkpoint:
            self.policy_value_net.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.policy_value_net.load_state_dict(checkpoint)
        
        self.policy_value_net.eval()
        self.mcts = MCTS(self.policy_value_net, c_puct=1.0, n_playout=100)

        self.player_id = player_id  # 记录AI的玩家编号（1或2）
        self.board_size = board_size  # 保存棋盘大小供外部使用
    
    def get_move(self, game):
        moves, probs = self.mcts.get_move_probs(game, temp=0)
        print("MCTS搜索后策略:", probs)
        if not moves:
            return None
        max_prob_idx = np.argmax(probs)
        return moves[max_prob_idx] if moves else None
    
    def update_with_move(self, move):
        self.mcts.update_with_move(move)

def human_vs_ai():
    # 加载模型以获取棋盘参数
    try:
        # 尝试加载纯自我对弈训练的最佳模型
        model_path = "pure_selfplay_checkpoint/best_model.pth"
        if not torch.load(model_path, map_location=torch.device('cpu')):
            raise FileNotFoundError
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        board_size = checkpoint.get('board_size', 3)
        win_length = 3  # 井字棋固定为3
    except FileNotFoundError:
        print("Error: 未找到模型文件 pure_selfplay_checkpoint/best_model.pth")
        print("请先运行 python pure_selfplay_trainer.py 训练模型")
        return
    
    # 初始化游戏，传入棋盘大小和获胜条件
    game = Game(board_size=board_size, win_length=win_length)
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
    ai = AIPlayer(model_path, ai_player)
    
    while True:
        game.render()
        if current_player == human_player:
            # 人类落子
            legal_moves = game.get_legal_moves()
            print(f"合法落子格式: 行 列 (0-{board_size-1}之间的整数)")
            
            try:
                user_input = input("你的选择: ").strip().lower()
                # 检查是否退出
                if user_input == 'q':
                    print("游戏已退出")
                    return
                # 直接使用行列坐标输入
                row, col = map(int, user_input.split())
                # 验证输入范围
                if row < 0 or row >= board_size or col < 0 or col >= board_size:
                    raise ValueError("坐标超出范围")
                move = (row, col)
                if move not in [tuple(m) for m in legal_moves]:
                    raise ValueError("无效坐标")
            except (ValueError):
                print(f"输入格式错误，请输入两个0-{board_size-1}之间的整数，用空格分隔")
                continue    
            # 执行人类落子
            game.make_move(move)
            # 人类落子后，通知AI的MCTS更新根节点
            ai.update_with_move(move)

            # 检查人类是否获胜
            if game.check_win(*move):
                game.render()
                print("恭喜，你赢了！")
                break
            # 切换到AI
            current_player = ai_player
        else:
            # AI落子
            move = ai.get_move(game)
            if not move:
                break  # 棋盘满了
            print(f"AI落子: {move}")
            game.make_move(move)
            ai.update_with_move(move)  # 新增
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
    human_vs_ai()