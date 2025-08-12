import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
import json

# 假设以下依赖模块已正确实现：
# - Game类：包含井字棋核心逻辑（get_state/make_move/check_win/is_full等）
# - PolicyValueNet类：策略价值网络（输入状态，输出动作概率和价值估计）
# - MCTS类：蒙特卡洛树搜索（基于策略价值网络生成动作概率）
from game import Game
from model import PolicyValueNet
from mcts import MCTS

# 配置参数
BOARD_SIZE = 3
WIN_LENGTH = 3
# 神经网络配置
RES_BLOCKS = 1  # 残差块数量
CHANNELS = [16, 32]  # 各层通道数，长度应等于残差块数量+1
# 训练数据保存路径
TRAINING_DATA_PATH = "checkpoint/training_data.npz"

def save_training_data(replay_buffer, file_path):
    """保存训练数据到文件"""
    states = np.array([item[0] for item in replay_buffer], dtype=np.float32)
    probs = np.array([item[1] for item in replay_buffer], dtype=np.float32)
    rewards = np.array([item[2] for item in replay_buffer], dtype=np.float32)
    np.savez(file_path, states=states, probs=probs, rewards=rewards)
    print(f"已保存 {len(replay_buffer)} 条训练数据至 {file_path}，文件大小: {os.path.getsize(file_path)/1024/1024:.2f} MB")

def load_training_data(file_path):
    """从文件加载训练数据"""
    if not os.path.exists(file_path):
        print(f"训练数据文件 {file_path} 不存在，将使用空缓存")
        return []
    try:
        data = np.load(file_path)
        states = data['states']
        probs = data['probs']
        rewards = data['rewards']
        replay_buffer = [(states[i], probs[i], rewards[i]) for i in range(len(states))]
        print(f"已加载 {len(replay_buffer)} 条训练数据")
        return replay_buffer
    except Exception as e:
        print(f"加载训练数据出错: {e}")
        return []

def train():
    replay_buffer = []  # 存储格式：(state, probs, reward)，其中reward基于当前玩家视角
    buffer_size = 20000  # 回放缓存大小
    batch_size = 64      # 训练批次大小
    game_batch_num = 5000  # 总训练轮次
    update_epochs = 10    # 每N轮自我对弈后更新一次模型
    checkpoint_freq = 100  # 模型保存频率
    c_puct = 1.0
    tau = 1.0
    n_playout = 100


    os.makedirs("checkpoint", exist_ok=True)  # 确保检查点目录存在
    
    # 检查训练进度，支持断点续训
    progress_file = "checkpoint/training_progress.json"
    start_batch = 0
    model_path = None
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f) 
            start_batch = progress.get('last_batch', 0)
            if start_batch > 0:
                model_path = f"checkpoint/temp_model_{start_batch}.pth"
                print(f"发现历史进度，将从模型 {model_path} 继续训练")
            print(f"从批次 {start_batch} 开始训练")
    
    # 初始化游戏、模型、优化器
    game = Game(board_size=BOARD_SIZE, win_length=WIN_LENGTH)
    policy_value_net = PolicyValueNet(board_size=BOARD_SIZE, num_res_blocks=RES_BLOCKS, channels=CHANNELS)
    optimizer = Adam(policy_value_net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.7)  # 学习率衰减
    
    # 加载历史模型和训练数据（如有）
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        policy_value_net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"已加载模型、优化器和调度器状态")
        # 加载训练数据
        replay_buffer = load_training_data(TRAINING_DATA_PATH)
    
    # 调整续训时的学习率
    if start_batch > 0:
        steps_to_skip = start_batch // update_epochs
        for _ in range(steps_to_skip):
            scheduler.step()
        print(f"续训时调整学习率为: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 主训练循环
    print(f"初始回放缓冲区大小: {len(replay_buffer)}")
    for i in range(start_batch, game_batch_num):
        game.reset()
        mcts = MCTS(policy_value_net, c_puct=c_puct, n_playout=n_playout)  # MCTS搜索次数

        play_data = []  # 存储单局数据：(state, current_player, move, probs)
        winner = None   # 最终胜负：1（黑胜）/2（白胜）/0（平局）
        
        while True:
            state = game.get_state()  # 获取当前棋盘状态（需返回可序列化格式）
            current_player = game.current_player  # 当前落子方（1或2）
            moves, probs = mcts.get_move_probs(game, temp=tau)  # 获取合法动作及概率

            
            if not moves:  # 无合法动作
                break
            
            # 处理概率（确保归一化，避免NaN/Inf）
            probs = np.asarray(probs)
            probs[~np.isfinite(probs)] = 0.0  # 替换无效值
            total = probs.sum()
            if total <= 0:
                probs = np.ones_like(probs) / len(probs)  # 均匀分布兜底
            else:
                probs = probs / total  # 归一化
            
            # 基于概率选择动作
            move = random.choices(moves, weights=probs)[0]
            
            # 转换为board_size×board_size维概率向量
            probs_board = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
            for m, p in zip(moves, probs):
                idx = m[0] * BOARD_SIZE + m[1]  # (row, col) -> 索引
                probs_board[idx] = p
            
            # 记录单步数据（含当前玩家）
            play_data.append((state, current_player, move, probs_board))
            
            # 执行动作并检查结局
            game.make_move(move)
            if game.check_win(*move):  # 检查当前动作是否导致获胜
                winner = current_player  # 获胜方为当前玩家
                break
            if game.is_full():  # 棋盘下满（平局）
                winner = 0
                break
            mcts.update_with_move(move)  # MCTS更新当前动作
        
        # 每局结束后打印缓冲区大小
        # if (i * num_games_per_batch + game_idx + 1) % 10 == 0:
        #     print(f"已完成 {i * num_games_per_batch + game_idx + 1} 局游戏，回放缓冲区大小: {len(replay_buffer)}")

        # 将单局数据转换为训练样本（带立场化奖励）并加入回放缓存
        for state, current_player, move, probs in play_data:
            # 计算立场化奖励：当前玩家获胜则+1，失败则-1，平局则0
            if winner == 0:
                reward = 0.0
            elif winner == current_player:
                reward = 1.0  # 当前玩家获胜
            else:
                reward = -1.0  # 当前玩家失败
            replay_buffer.append((state, probs, reward))

        # 控制缓存大小（超出则删除最早数据）
        if len(replay_buffer) > buffer_size:
            # 计算需要删除的数据量
            num_to_remove = len(replay_buffer) - buffer_size
            replay_buffer = replay_buffer[num_to_remove:]
            print(f"回放缓冲区超过最大容量，删除最早的 {num_to_remove} 条数据")
        
        # 控制缓存大小（超出则删除最早数据）
        # if len(replay_buffer) > buffer_size:
        #     replay_buffer.pop(0)
        
        # 每update_epochs轮迭代更新一次模型
        if i % update_epochs == 0 and i != 0 and len(replay_buffer) >= batch_size:
            # 从回放缓存采样（使用优先级采样，无平滑处理）
            # 1. 计算所有样本的价值误差（预测值与真实奖励的差距）
            with torch.no_grad():
                # 批量转换状态为张量
                states_np = np.array([s[0] for s in replay_buffer], dtype=np.float32)
                state_tensor = torch.FloatTensor(states_np).to(policy_value_net.device)
                # 模型预测价值
                _, values = policy_value_net(state_tensor)
                values_np = values.cpu().numpy().flatten()  # 转换为numpy数组
            
            # 2. 计算优先级（基于绝对误差，无平滑）
            rewards_np = np.array([s[2] for s in replay_buffer], dtype=np.float32)
            value_errors = np.abs(values_np - rewards_np) + 1e-6  # 误差（加小值避免为0）
            priorities = value_errors  # 直接用误差作为优先级（无平滑/归一化）
            priorities = priorities / np.sum(priorities)  # 归一化概率分布
            
            # 3. 基于优先级采样
            sample_indices = np.random.choice(len(replay_buffer), batch_size, p=priorities)
            # 提取采样后的批次数据
            state_batch = np.array([replay_buffer[idx][0] for idx in sample_indices], dtype=np.float32)
            probs_batch = np.array([replay_buffer[idx][1] for idx in sample_indices], dtype=np.float32)
            reward_batch = np.array([replay_buffer[idx][2] for idx in sample_indices], dtype=np.float32)
            
            # 4. 转换为张量并训练
            state_tensor = torch.FloatTensor(state_batch).to(policy_value_net.device)
            probs_tensor = torch.FloatTensor(probs_batch).to(policy_value_net.device)
            reward_tensor = torch.FloatTensor(reward_batch).to(policy_value_net.device)
            
            # 计算损失
            optimizer.zero_grad()
            policy, value = policy_value_net(state_tensor)
            # 策略损失（交叉熵）
            policy_loss = -torch.mean(torch.sum(probs_tensor * policy, dim=1))
            # 价值损失（MSE）
            value_loss = F.mse_loss(value.view(-1), reward_tensor)
            # 总损失（价值损失加权）
            total_loss = policy_loss + 1.5 * value_loss
            
            # 反向传播与参数更新
            total_loss.backward()
            optimizer.step()
            scheduler.step()  # 在PyTorch中，应在optimizer.step()后调用scheduler.step()
            
            # 打印训练信息
            print(f"批次 {i}, 学习率: {optimizer.param_groups[0]['lr']:.6f}, "
                  f"总损失: {total_loss.item():.4f}, "
                  f"策略损失: {policy_loss.item():.4f}, "
                  f"价值损失: {value_loss.item():.4f}")
        
        # 定期保存模型并对比性能
        if i % checkpoint_freq == 0 and i != 0:
            temp_model_path = f"checkpoint/temp_model_{i}.pth"
            torch.save({
                'model_state_dict': policy_value_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'board_size': BOARD_SIZE,
                'win_length': WIN_LENGTH,
                'res_blocks': RES_BLOCKS,
                'channels': CHANNELS
            }, temp_model_path)
            
            # 与上一版本模型对比
            # prev_model_path = f"checkpoint/model_{i - checkpoint_freq}.pth"
            # if os.path.exists(prev_model_path):
            #     print(f"对比模型（批次 {i} vs {i - checkpoint_freq}）...")
            #     current_win_rate = compare2_models(temp_model_path, prev_model_path)
            #     print(f"当前模型胜率: {current_win_rate:.2f}%"
                
            #     if current_win_rate >= 50:
            #         os.rename(temp_model_path, f"checkpoint/model_{i}.pth")
            #         print(f"当前模型更优，保存为 model_{i}.pth")
            #     else:
            #         os.remove(temp_model_path)
            #         print(f"保留上一版本模型，丢弃当前模型")
            #         # 加载上一版本模型继续训练
            #         checkpoint = torch.load(prev_model_path)
            #         policy_value_net.load_state_dict(checkpoint['model_state_dict'])
            #         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # else:
            #     # 无历史模型，直接保存
            #     os.rename(temp_model_path, f"checkpoint/model_{i}.pth")
            #     print(f"无历史模型，保存当前模型为 model_{i}.pth")
            
            # 保存训练进度
            with open(progress_file, 'w') as f:
                json.dump({'last_batch': i}, f)
            print(f"已保存训练进度至 {progress_file}")
            
            # 保存训练数据
            save_training_data(replay_buffer, TRAINING_DATA_PATH)
    
    # 训练结束，保存最终模型和数据
    torch.save({
        'model_state_dict': policy_value_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'board_size': BOARD_SIZE,
        'win_length': WIN_LENGTH,
        'res_blocks': RES_BLOCKS,
        'channels': CHANNELS
    }, "checkpoint/final_model.pth")
    with open(progress_file, 'w') as f:
        json.dump({'last_batch': game_batch_num}, f)
    # 保存最终训练数据
    save_training_data(replay_buffer, TRAINING_DATA_PATH)
    print("训练结束，已保存最终模型、进度和训练数据")


def compare_models(model_path1, model_path2, num_games=10):
    """对比两个模型的胜率（轮换先手确保公平性）"""
    game = Game()
    # 加载两个模型
    net1 = PolicyValueNet()
    net2 = PolicyValueNet()
    checkpoint1 = torch.load(model_path1)
    checkpoint2 = torch.load(model_path2)
    net1.load_state_dict(checkpoint1['model_state_dict'])
    net2.load_state_dict(checkpoint2['model_state_dict'])
    net1.eval()
    net2.eval()
    
    # 创建MCTS玩家
    mcts1 = MCTS(net1, n_playout=200)
    mcts2 = MCTS(net2, n_playout=200)
    
    wins1 = 0  # 模型1胜利次数
    wins2 = 0  # 模型2胜利次数
    draws = 0  # 平局次数
    
    for i in range(num_games):
        # print(f"对比对局 {i+1}/{num_games}...")
        game.reset()
        # 轮换先手（偶数局模型1先手，奇数局模型2先手）
        if i % 2 == 0:
            player1, player2 = 1, 2  # 模型1=黑棋(1)，模型2=白棋(2)
        else:
            player1, player2 = 2, 1  # 模型1=白棋(2)，模型2=黑棋(1)
        
        current_player = 1  # 当前落子方（1=黑，2=白）
        mcts1.update_with_move(-1)  # 重置MCTS
        mcts2.update_with_move(-1)
        
        while True:
            if current_player == player1:
                # 模型1落子（贪心选择最高概率动作）
                moves, probs = mcts1.get_move_probs(game, temp=0)
                move = moves[np.argmax(probs)]
                mcts1.update_with_move(move)
                mcts2.update_with_move(move)
            else:
                # 模型2落子
                moves, probs = mcts2.get_move_probs(game, temp=0)
                move = moves[np.argmax(probs)]
                mcts1.update_with_move(move)
                mcts2.update_with_move(move)
            
            game.make_move(move)
            # 检查胜负
            if game.check_win(*move):
                if current_player == player1:
                    wins1 += 1
                else:
                    wins2 += 1
                break
            if game.is_full():
                draws += 1
                break
            # 切换玩家
            current_player = 2 if current_player == 1 else 1
    
    # 计算胜率（平局按0.5胜计算）
    total = wins1 + wins2 + draws
    win_rate = (wins1 + 0.5 * draws) / total * 100 if total > 0 else 50.0
    return win_rate

def compare2_models(model_path1, model_path2, num_games=10):
    """对比两个模型的胜率（无MCTS，直接用模型策略输出，轮换先手）"""
    game = Game()
    # 加载两个模型
    net1 = PolicyValueNet()
    net2 = PolicyValueNet()
    checkpoint1 = torch.load(model_path1)
    checkpoint2 = torch.load(model_path2)
    net1.load_state_dict(checkpoint1['model_state_dict'])
    net2.load_state_dict(checkpoint2['model_state_dict'])
    net1.eval()  # 评估模式，关闭dropout等
    net2.eval()
    device = next(net1.parameters()).device  # 获取模型所在设备（CPU/GPU）
    
    wins1 = 0  # 模型1胜利次数
    wins2 = 0  # 模型2胜利次数
    draws = 0  # 平局次数
    
    for i in range(num_games):
        # print(f"对比对局 {i+1}/{num_games}...")
        game.reset()
        # 轮换先手（偶数局模型1先手，奇数局模型2先手）
        if i % 2 == 0:
            # 模型1执黑（1），模型2执白（2）
            model_black, model_white = net1, net2
        else:
            # 模型2执黑（1），模型1执白（2）
            model_black, model_white = net2, net1
        
        current_player = 1  # 当前落子方（1=黑，2=白）
        
        while True:
            # 获取当前棋盘状态（需转换为模型输入格式，如3x3数组）
            state = game.get_state()  # 假设返回格式为 (3,3) 或类似
            # 转换为模型输入张量（增加批次维度+设备转换）
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # 根据当前玩家选择模型
            if current_player == 1:
                # 黑棋落子（用model_black）
                policy, _ = model_black(state_tensor)  # 仅用策略输出（9维概率）
            else:
                # 白棋落子（用model_white）
                policy, _ = model_white(state_tensor)
            
            # 解析策略：获取合法动作并选择概率最高的
            policy_np = policy.squeeze().cpu().detach().numpy()  # 转为numpy（9维）
            legal_moves = game.get_legal_moves()  # 假设返回合法落子列表，格式如[(0,0), (0,1), ...]
            
            if not legal_moves:
                # 无合法动作（井字棋理论上不会出现）
                break
            
            # 将合法动作转换为9维索引，筛选对应概率
            legal_indices = [move[0] * 3 + move[1] for move in legal_moves]
            legal_probs = [policy_np[idx] for idx in legal_indices]
            
            # 贪心选择概率最高的合法动作
            max_idx = np.argmax(legal_probs)
            move = legal_moves[max_idx]  # 选中的落子位置
            
            # 执行落子
            game.make_move(move)
            
            # 检查胜负
            if game.check_win(*move):
                if (current_player == 1 and model_black is net1) or (current_player == 2 and model_white is net1):
                    # 模型1获胜
                    wins1 += 1
                else:
                    # 模型2获胜
                    wins2 += 1
                break
            if game.is_full():
                # 平局
                draws += 1
                break
            
            # 切换玩家
            current_player = 2 if current_player == 1 else 1
    
    # 计算胜率（平局按0.6胜计算，鼓励接近最优的模型）
    total = wins1 + wins2 + draws
    win_rate = (wins1 + 0.6 * draws) / total * 100 if total > 0 else 50.0
    return win_rate

if __name__ == "__main__":
    train()