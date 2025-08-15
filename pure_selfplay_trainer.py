#!/usr/bin/env python3
"""
纯自我对弈训练器 - 无任何专家知识注入
解决回退问题的通用方案
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import os
import json

from game import Game
from model import PolicyValueNet
from mcts import MCTS

class PureSelfPlayTrainer:
    """纯自我对弈训练器 - 无专家系统"""
    
    def __init__(self):
        # 配置参数
        self.board_size = 3
        self.input_channels = 64      # 第一层卷积通道数（恢复成功配置）
        self.res_channels = [32, 64]  # 残差块通道数数组（恢复成功配置）
        self.dropout_rate = 0.0
        
        # 训练超参数 - 关键：稳定的配置
        self.training_rounds = 1000
        self.mcts_simulations = 100
        self.learning_rate = 0.002  # 初始学习率
        self.lr_decay_step = 100    # 学习率衰减步长
        self.lr_decay_gamma = 0.9   # 学习率衰减因子
        self.batch_size = 64        # 更大批次，稳定梯度
        self.buffer_size = 2000     # 更大缓冲区，保持多样性
        
        # 🔑 关键改进：动态温度和探索策略
        self.initial_temperature = 2.0  # 高初始温度，强制探索
        self.final_temperature = 0.1    # 低最终温度，收敛策略
        self.temperature_decay_rounds = 600  # 温度衰减轮数
        
        # 🔑 关键改进：渐进式训练强度
        self.initial_games_per_round = 2    # 初期少量游戏
        self.final_games_per_round = 8      # 后期更多游戏
        self.game_ramp_rounds = 300         # 游戏数量爬坡轮数
        
        # 🔑 关键改进：自适应MCTS参数
        self.initial_c_puct = 2.0   # 高探索
        self.final_c_puct = 1.0     # 标准探索
        
        # 监控和早停
        self.evaluation_interval = 50
        self.patience = 200  # 更长耐心，避免过早停止
        self.min_improvement = 0.01  # 最小改善阈值
        
        # 性能追踪
        self.performance_history = []
        
    def get_dynamic_temperature(self, round_num):
        """动态温度调度"""
        if round_num <= self.temperature_decay_rounds:
            progress = round_num / self.temperature_decay_rounds
            temp = self.initial_temperature * (1 - progress) + self.final_temperature * progress
        else:
            temp = self.final_temperature
        return max(temp, 0.1)  # 最小温度保护
    
    def get_dynamic_games_count(self, round_num):
        """动态游戏数量"""
        if round_num <= self.game_ramp_rounds:
            progress = round_num / self.game_ramp_rounds
            games = int(self.initial_games_per_round * (1 - progress) + 
                       self.final_games_per_round * progress)
        else:
            games = self.final_games_per_round
        return max(games, 2)  # 至少2局游戏
    
    def get_dynamic_c_puct(self, round_num):
        """动态探索参数 - 平滑衰减，避免突然下降"""
        # 🔑 关键修复：延长衰减期，避免轮次300的突然下降
        decay_rounds = 600  # 延长到600轮，与温度衰减一致
        if round_num <= decay_rounds:
            progress = round_num / decay_rounds
            # 使用平滑的指数衰减而非线性衰减
            c_puct = self.initial_c_puct * (self.final_c_puct / self.initial_c_puct) ** progress
        else:
            c_puct = self.final_c_puct
        return max(c_puct, 1.0)  # 确保最小值为1.0
    
    def evaluate_model(self, net, round_num):
        """评估模型性能 - 纯粹基于策略分析"""
        game = Game()
        
        # 使用更高温度评估，避免MCTS过度放大网络偏好
        mcts = MCTS(net, c_puct=1.0, n_playout=100)
        moves, probs = mcts.get_move_probs(game, temp=1.0)  # 修复：使用温度1.0
        
        # 计算中心偏好
        center_prob = 0
        corner_probs = []
        edge_probs = []
        
        for move, prob in zip(moves, probs):
            if move == (1, 1):  # 中心
                center_prob = prob
            elif move in [(0, 0), (0, 2), (2, 0), (2, 2)]:  # 角落
                corner_probs.append(prob)
            else:  # 边缘
                edge_probs.append(prob)
        
        # 修复：避免极小概率导致的天文数字比值
        avg_corner = max(np.mean(corner_probs) if corner_probs else 0.0001, 1e-6)
        avg_edge = max(np.mean(edge_probs) if edge_probs else 0.0001, 1e-6)
        
        # 使用安全的比值计算，避免除零和天文数字
        center_corner_ratio = min(center_prob / avg_corner, 1000.0)  # 限制最大比值
        center_edge_ratio = min(center_prob / avg_edge, 1000.0)      # 限制最大比值
        
        # 综合评分：既要中心偏好，也要合理的边缘策略
        # 理想的井字棋策略：中心 > 角落 > 边缘
        ideal_score = center_corner_ratio * 0.7 + center_edge_ratio * 0.3
        
        return {
            'center_prob': center_prob,
            'corner_prob': avg_corner,
            'edge_prob': avg_edge,
            'center_corner_ratio': center_corner_ratio,
            'center_edge_ratio': center_edge_ratio,
            'ideal_score': ideal_score,
            'round': round_num
        }
    
    def self_play_game(self, net, round_num):
        """单局自我对弈"""
        game = Game()
        
        # 动态参数
        temperature = self.get_dynamic_temperature(round_num)
        c_puct = self.get_dynamic_c_puct(round_num)
        
        mcts = MCTS(net, c_puct=c_puct, n_playout=self.mcts_simulations)
        
        states, probs, rewards = [], [], []
        
        # 游戏主循环
        winner = None
        while True:
            if game.is_full():
                winner = 0  # 平局
                break
            
            # 获取MCTS策略
            state = game.get_state()
            moves, move_probs = mcts.get_move_probs(game, temp=temperature)
            
            # 记录训练数据
            states.append(state.copy())
            prob_array = np.zeros(9)
            for move, prob in zip(moves, move_probs):
                move_idx = move[0] * 3 + move[1] if isinstance(move, tuple) else move
                prob_array[move_idx] = prob
            probs.append(prob_array)
            
            # 选择并执行动作
            action_idx = np.random.choice(len(moves), p=move_probs)
            selected_move = moves[action_idx]
            
            is_win = game.make_move(selected_move)
            if is_win:
                winner = 2 if game.current_player == 1 else 1
                break
            
            # 更新MCTS状态
            update_move = (selected_move[0] * 3 + selected_move[1] 
                          if isinstance(selected_move, tuple) else selected_move)
            mcts.update_with_move(update_move)
        
        # 计算每个位置的奖励
        game_data = []
        for i, (state, prob) in enumerate(zip(states, probs)):
            current_player = 1 if i % 2 == 0 else -1
            if winner == 0:
                reward = 0.0  # 平局
            elif winner == current_player:
                reward = 1.0  # 获胜
            else:
                reward = -1.0  # 失败
            
            game_data.append((state, prob, reward))
        
        return game_data
    
    def train(self):
        """主训练循环"""
        print("🎯 纯自我对弈训练器 - 无专家知识")
        print("="*60)
        print(f"🧠 网络架构: input_channels={self.input_channels}, res_channels={self.res_channels}")
        print(f"🔍 MCTS搜索: {self.mcts_simulations}次")
        print(f"🌡️ 温度范围: {self.initial_temperature} → {self.final_temperature}")
        print(f"🎮 游戏数量: {self.initial_games_per_round} → {self.final_games_per_round}")
        print(f"⚖️ 探索参数: {self.initial_c_puct} → {self.final_c_puct}")
        print(f"💾 缓冲区大小: {self.buffer_size}")
        print("")
        
        # 初始化网络和优化器
        net = PolicyValueNet(
            board_size=self.board_size,
            input_channels=self.input_channels,
            res_channels=self.res_channels,
            dropout_rate=self.dropout_rate
        )
        
        optimizer = Adam(net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # 🔑 关键修复：添加学习率调度器，防止后期过拟合
        scheduler = StepLR(optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_gamma)
        replay_buffer = []
        
        # 创建输出目录
        os.makedirs("pure_selfplay_checkpoint", exist_ok=True)
        
        # 训练状态追踪
        best_score = 0
        best_round = 0
        no_improvement_count = 0
        
        # 主训练循环
        for round_num in range(1, self.training_rounds + 1):
            current_temp = self.get_dynamic_temperature(round_num)
            current_games = self.get_dynamic_games_count(round_num)
            current_c_puct = self.get_dynamic_c_puct(round_num)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"轮次 {round_num}/{self.training_rounds} "
                  f"(T={current_temp:.2f}, 游戏={current_games}, C={current_c_puct:.2f}, LR={current_lr:.5f})")
            
            # 1. 自我对弈生成数据
            round_data = []
            for game_idx in range(current_games):
                game_data = self.self_play_game(net, round_num)
                round_data.extend(game_data)
            
            # 2. 更新回放缓冲区
            replay_buffer.extend(round_data)
            if len(replay_buffer) > self.buffer_size:
                # 🔑 关键：保持缓冲区多样性，随机移除而非FIFO
                excess = len(replay_buffer) - self.buffer_size
                indices_to_remove = random.sample(range(len(replay_buffer)), excess)
                replay_buffer = [replay_buffer[i] for i in range(len(replay_buffer)) 
                               if i not in indices_to_remove]
            
            # 3. 网络训练
            if len(replay_buffer) >= self.batch_size:
                # 多次训练以提高学习效率
                train_iterations = max(1, len(round_data) // 16)  # 自适应训练次数
                
                for _ in range(train_iterations):
                    batch = random.sample(replay_buffer, self.batch_size)
                    
                    states = torch.FloatTensor([item[0] for item in batch]).to(net.device)
                    target_probs = torch.FloatTensor([item[1] for item in batch]).to(net.device)
                    target_values = torch.FloatTensor([item[2] for item in batch]).to(net.device)
                    
                    # 前向传播
                    log_probs, values = net(states)
                    
                    # 损失计算
                    value_loss = F.mse_loss(values.squeeze(), target_values)
                    policy_loss = -torch.mean(torch.sum(target_probs * log_probs, dim=1))
                    total_loss = value_loss + policy_loss
                    
                    # 反向传播
                    optimizer.zero_grad()
                    total_loss.backward()
                    # 梯度裁剪，稳定训练
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
            
            # 4. 定期评估
            if round_num % self.evaluation_interval == 0:
                performance = self.evaluate_model(net, round_num)
                self.performance_history.append(performance)
                
                print(f"  📊 评估结果:")
                print(f"    中心概率: {performance['center_prob']:.4f}")
                print(f"    中心/角落比值: {performance['center_corner_ratio']:.3f}")
                print(f"    综合评分: {performance['ideal_score']:.3f}")
                
                # 早停检查
                if performance['ideal_score'] > best_score + self.min_improvement:
                    best_score = performance['ideal_score']
                    best_round = round_num
                    no_improvement_count = 0
                    
                    # 保存最佳模型
                    best_model_path = "pure_selfplay_checkpoint/best_model.pth"
                    torch.save({
                        'model_state_dict': net.state_dict(),
                        'round': round_num,
                        'performance': performance,
                        'board_size': self.board_size,
                        'input_channels': self.input_channels,
                        'res_channels': self.res_channels,
                        'dropout_rate': self.dropout_rate,
                        'training_config': {
                            'temperature_range': (self.initial_temperature, self.final_temperature),
                            'c_puct_range': (self.initial_c_puct, self.final_c_puct)
                        }
                    }, best_model_path)
                    print(f"    🏆 新最佳模型! 评分: {best_score:.3f}")
                else:
                    no_improvement_count += self.evaluation_interval
                
                # 定期保存
                if round_num % (self.evaluation_interval * 4) == 0:
                    model_path = f"pure_selfplay_checkpoint/model_{round_num}.pth"
                    torch.save({
                        'model_state_dict': net.state_dict(),
                        'round': round_num,
                        'performance': performance
                    }, model_path)
                    print(f"    💾 已保存检查点: {model_path}")
                
                # 早停检查
                if no_improvement_count >= self.patience:
                    print(f"  ⏹️ 早停: {self.patience}轮无显著改善")
                    break
            
            # 🔑 关键修复：学习率调度，防止后期过拟合
            scheduler.step()
        
        # 保存训练历史
        history_path = "pure_selfplay_checkpoint/training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
        
        print(f"✅ 纯自我对弈训练完成!")
        print(f"🏆 最佳模型: 轮次{best_round}, 评分{best_score:.3f}")
        print(f"📊 训练历史已保存到: {history_path}")

if __name__ == "__main__":
    trainer = PureSelfPlayTrainer()
    trainer.train()
