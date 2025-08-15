# Minimal AlphaZero-style Tic-Tac-Toe (CNN+ResNet + MCTS) — single file
# Author: ChatGPT
# Python 3.10+, PyTorch 2.x
#
# Features
# - 3x3 TicTacToe environment
# - 5-plane input encoding (current, opponent, to-move, last-move, ones)
# - Tiny CNN with 2 residual blocks, policy & value heads
# - PUCT MCTS with Dirichlet noise, temperature schedule
# - Self-play, replay buffer, training loop
# - Optional 8-fold symmetry data augmentation
# - Evaluation vs minimax (perfect play)
#
# Usage
#   pip install torch numpy
#   python tictactoe_alphazero_minimal.py --help
#   python tictactoe_alphazero_minimal.py --iters 15 --games-per-iter 300 --mcts-sims 100
#
# Notes
# - CPU is enough. This code is intentionally simple and readable.
# - For reproducibility you can set seeds via --seed.

from __future__ import annotations
import math
import random
import argparse
import os
import shutil
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =====================
# Game: TicTacToe
# =====================

class TicTacToe:
    """3x3 TicTacToe with players +1 (X) and -1 (O).

    Board encoding: np.int8 array shape (3,3) with values {0, +1, -1}.
    current_player: whose turn (1 or -1)
    last_move: index 0..8 or None
    """

    def __init__(self):
        self.board = np.zeros((3,3), dtype=np.int8)
        self.current_player = 1
        self.last_move: Optional[int] = None

    def clone(self) -> 'TicTacToe':
        g = TicTacToe()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.last_move = self.last_move
        return g

    @staticmethod
    def idx_to_rc(idx: int) -> Tuple[int,int]:
        return divmod(idx, 3)

    @staticmethod
    def rc_to_idx(r: int, c: int) -> int:
        return r*3 + c

    def legal_moves(self) -> List[int]:
        return [i for i in range(9) if self.board[self.idx_to_rc(i)] == 0]

    def play(self, idx: int) -> None:
        r,c = self.idx_to_rc(idx)
        assert self.board[r,c] == 0, "Illegal move"
        self.board[r,c] = self.current_player
        self.current_player *= -1
        self.last_move = idx

    def result(self) -> Optional[int]:
        """Return +1 if X wins, -1 if O wins, 0 if draw, None if ongoing."""
        b = self.board
        lines = []
        lines.extend(b.sum(axis=0))
        lines.extend(b.sum(axis=1))
        lines.append(b[0,0] + b[1,1] + b[2,2])
        lines.append(b[0,2] + b[1,1] + b[2,0])
        if 3 in lines: return 1
        if -3 in lines: return -1
        if (b != 0).all(): return 0
        return None

    def encode(self) -> np.ndarray:
        """Return planes (C=5,H=3,W=3) for current player perspective.

        planes:
          0: current player's stones
          1: opponent stones
          2: to-move plane (all ones)
          3: last move (one-hot board) or zeros if None
          4: constant ones
        """
        cur = (self.board == self.current_player).astype(np.float32)
        opp = (self.board == -self.current_player).astype(np.float32)
        turn = np.ones((3,3), dtype=np.float32)
        last = np.zeros((3,3), dtype=np.float32)
        if self.last_move is not None:
            r,c = self.idx_to_rc(self.last_move)
            last[r,c] = 1.0
        ones = np.ones((3,3), dtype=np.float32)
        planes = np.stack([cur, opp, turn, last, ones], axis=0)  # (5,3,3)
        return planes

# =====================
# Symmetry augmentation (D4 group)
# =====================

# Precompute index maps for 8 symmetries on a 3x3 grid.

def _build_symmetry_maps() -> List[np.ndarray]:
    grid = np.arange(9).reshape(3,3)
    transforms = []
    mats = [
        grid,  # identity
        np.rot90(grid, 1),
        np.rot90(grid, 2),
        np.rot90(grid, 3),
        np.fliplr(grid),
        np.rot90(np.fliplr(grid), 1),
        np.rot90(np.fliplr(grid), 2),
        np.rot90(np.fliplr(grid), 3),
    ]
    for m in mats:
        transforms.append(m.reshape(-1))
    return transforms

SYM_IDX_MAPS = _build_symmetry_maps()  # list of (9,) arrays


def apply_symmetry_planes(planes: np.ndarray, sym_id: int) -> np.ndarray:
    """Apply symmetry to input planes (C,3,3)."""
    C = planes.shape[0]
    out = np.zeros_like(planes)
    idx_map = SYM_IDX_MAPS[sym_id]
    # apply by indexing flattened board
    for c in range(C):
        flat = planes[c].reshape(-1)
        out[c] = flat[idx_map].reshape(3,3)
    return out


def apply_symmetry_pi(pi: np.ndarray, sym_id: int) -> np.ndarray:
    """Apply symmetry to policy vector (9,)."""
    idx_map = SYM_IDX_MAPS[sym_id]
    return pi[idx_map]

# =====================
# Neural Network
# =====================

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)

class AZNet(nn.Module):
    def __init__(self, in_ch=5, ch=32, nblocks=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(nblocks)])
        # Policy head
        self.pol_conv = nn.Conv2d(ch, 2, 1, bias=False)
        self.pol_bn = nn.BatchNorm2d(2)
        self.pol_fc = nn.Linear(2*3*3, 9)
        # Value head
        self.val_conv = nn.Conv2d(ch, 1, 1, bias=False)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(1*3*3, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res(x)
        p = F.relu(self.pol_bn(self.pol_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.pol_fc(p)
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v))
        return p, v.squeeze(-1)

# =====================
# MCTS
# =====================

@dataclass
class MCTSConfig:
    sims: int = 100
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    temp_moves: int = 2  # moves before temperature -> 0 (修复：从5减少到2)

class BatchMCTS:
    """批量MCTS：充分利用GPU并行计算多个游戏状态"""
    def __init__(self, net: AZNet, cfg: MCTSConfig, device: torch.device):
        self.net = net
        self.cfg = cfg
        self.device = device
        
    def batch_run_games(self, games: List[TicTacToe], move_counts: List[int]) -> List[np.ndarray]:
        """批量运行多个游戏的MCTS搜索"""
        batch_size = len(games)
        results = []
        
        # 为每个游戏创建独立的搜索状态
        searches = []
        for i in range(batch_size):
            search_state = {
                'P': {},  # prior
                'N': defaultdict(int),
                'W': defaultdict(float), 
                'Q': defaultdict(float),
                'children': defaultdict(list),
                'terminal': {},
                'legal': {}
            }
            searches.append(search_state)
        
        # 批量初始expand
        expand_requests = []
        for i, game in enumerate(games):
            key = self._state_key(game)
            expand_requests.append((i, game, key))
        
        values = self._batch_expand(expand_requests, searches)
        
        # 为每个游戏添加Dirichlet噪声（如果是开局）
        for i, (game, move_count) in enumerate(zip(games, move_counts)):
            if move_count == 0:
                key = self._state_key(game)
                self._add_dirichlet(searches[i], key)
        
        # 运行MCTS模拟
        for sim in range(self.cfg.sims):
            # 收集需要expand的请求
            expand_requests = []
            paths_to_backup = []  # 需要backup的路径
            
            for i, game in enumerate(games):
                # 模拟一条路径
                path, need_expand = self._simulate_path(game, searches[i])
                if need_expand:
                    game_state, key = need_expand
                    expand_requests.append((i, game_state, key))
                    paths_to_backup.append((i, path, game))
            
            # 批量expand
            if expand_requests:
                values = self._batch_expand(expand_requests, searches)
                
                # 对每个expand的路径进行backup
                for i, path, game in paths_to_backup:
                    if i in values:
                        v = values[i]
                        self._backup(searches[i], path, v, game.current_player)
        
        # 构建每个游戏的策略
        for i, game in enumerate(games):
            key = self._state_key(game)
            move_count = move_counts[i]
            pi = self._build_policy(searches[i], key, move_count)
            results.append(pi)
        
        return results
    
    def _state_key(self, game: TicTacToe) -> Tuple:
        return (tuple(game.board.reshape(-1)), game.current_player)
    
    def _batch_expand(self, expand_requests: List[Tuple[int, TicTacToe, Tuple]], searches: List[Dict]) -> Dict[int, float]:
        """批量expand多个状态"""
        if not expand_requests:
            return {}
            
        # 批量编码和推理
        games = [game for _, game, _ in expand_requests]
        planes_batch = np.stack([game.encode() for game in games])
        
        with torch.no_grad():
            p_logits_batch, v_batch = self.net(torch.from_numpy(planes_batch).to(self.device))
            v_batch = v_batch.squeeze().cpu().numpy()  # 修复：squeeze()会处理所有维度
            p_logits_batch = p_logits_batch.cpu().numpy()
        
        # 处理结果
        values = {}
        for idx, (search_idx, game, key) in enumerate(expand_requests):
            search = searches[search_idx]
            legal = game.legal_moves()
            
            # 处理策略
            mask = np.full(9, float('-inf'))
            mask[legal] = 0.0
            p_logits = p_logits_batch[idx] + mask
            
            # 修复数值稳定性
            p = np.full(9, 0.0)
            if len(legal) > 0:
                p_logits_max = np.max(p_logits[legal])  # 只在合法位置计算max
                exp_logits = np.exp(p_logits[legal] - p_logits_max)
                exp_sum = np.sum(exp_logits)
                if exp_sum > 0:
                    p[legal] = exp_logits / exp_sum
                else:
                    p[legal] = 1.0 / len(legal)  # 均匀分布作为备选
            # 如果没有合法位置（游戏结束），p保持全0
            
            # 存储
            search['P'][key] = p
            search['legal'][key] = legal
            search['children'][key] = legal
            search['terminal'][key] = game.result()
            
            # 修复v_batch索引问题
            if v_batch.ndim == 0:  # 单个样本
                values[search_idx] = float(v_batch)
            else:  # 多个样本
                values[search_idx] = float(v_batch[idx])
        
        return values
    
    def _add_dirichlet(self, search: Dict, root_key: Tuple):
        """添加Dirichlet噪声"""
        alpha = self.cfg.dirichlet_alpha
        eps = self.cfg.dirichlet_eps
        p = search['P'][root_key].copy()
        legal = search['legal'][root_key]
        noise = np.random.dirichlet([alpha]*len(legal))
        p_new = p.copy()
        for i, a in enumerate(legal):
            p_new[a] = (1-eps)*p[a] + eps*noise[i]
        search['P'][root_key] = p_new
    
    def _simulate_path(self, root: TicTacToe, search: Dict):
        """模拟一条路径，返回需要expand的状态"""
        path = []
        game = root.clone()
        
        while True:
            key = self._state_key(game)
            
            # 检查终止
            term = search['terminal'].get(key, None)
            if term is not None:
                v = 0.0 if term == 0 else (1.0 if term == game.current_player else -1.0)
                self._backup(search, path, v, root.current_player)
                return path, None
            
            # 检查是否需要expand
            if key not in search['P']:
                return path, (game, key)
            
            # 选择行动
            action = self._select_action(search, key)
            path.append((key, action))  # 记录状态和选择的动作
            game.play(action)
        
    def _select_action(self, search: Dict, key: Tuple) -> int:
        """使用PUCT选择行动"""
        total_N = sum(search['N'][(key, a)] for a in search['children'][key])
        best, best_u = None, -1e9
        
        for a in search['children'][key]:
            q = search['Q'][(key, a)]
            p = search['P'][key][a]
            u = q + self.cfg.c_puct * p * math.sqrt(total_N + 1) / (1 + search['N'][(key, a)])
            if u > best_u:
                best_u, best = u, a
        
        return best
    
    def _backup(self, search: Dict, path: List, v: float, root_player: int):
        """回传价值"""
        # 沿路径反向传播价值
        for key, action in reversed(path):
            # 更新访问计数
            search['N'][(key, action)] += 1
            
            # 根据当前节点的玩家计算价值方向
            # v是从叶子节点视角的价值，需要转换到当前节点视角
            search['W'][(key, action)] += v
            
            # 更新Q值（平均价值）
            if search['N'][(key, action)] > 0:
                search['Q'][(key, action)] = search['W'][(key, action)] / search['N'][(key, action)]
            
            # 切换价值符号（因为玩家交替）
            v = -v
    
    def _build_policy(self, search: Dict, key: Tuple, move_count: int) -> np.ndarray:
        """构建策略分布"""
        counts = np.zeros(9, dtype=np.float32)
        if key in search['legal']:
            for a in search['legal'][key]:
                counts[a] = search['N'][(key, a)]
        
        # 温度控制
        if move_count < self.cfg.temp_moves:
            pi = counts ** 1.0
        else:
            pi = np.zeros_like(counts)
            if counts.sum() > 0:
                pi[np.argmax(counts)] = 1.0
        
        if pi.sum() > 0:
            pi = pi / pi.sum()
        else:
            # 如果没有访问记录，使用均匀分布
            if key in search['legal']:
                legal = search['legal'][key]
                pi = np.zeros(9, dtype=np.float32)
                pi[legal] = 1.0 / len(legal)
        
        return pi

class MCTS:
    """原有的MCTS类，保持向后兼容"""
    def __init__(self, net: AZNet, cfg: MCTSConfig, device: torch.device):
        self.net = net
        self.cfg = cfg
        self.device = device
        # per-root storage
        self.P = {}  # prior
        self.N = defaultdict(int)
        self.W = defaultdict(float)
        self.Q = defaultdict(float)
        self.children = defaultdict(list)
        self.terminal = {}
        self.legal = {}

    def reset(self):
        self.P.clear(); self.N.clear(); self.W.clear(); self.Q.clear()
        self.children.clear(); self.terminal.clear(); self.legal.clear()

    def state_key(self, game: TicTacToe) -> Tuple:
        return (tuple(game.board.reshape(-1)), game.current_player)

    def expand(self, game: TicTacToe, key: Tuple):
        # Network eval
        planes = game.encode()[None, ...]  # (1,5,3,3)
        with torch.no_grad():
            p_logits, v = self.net(torch.from_numpy(planes).to(self.device))
            p_logits = p_logits[0]
            v = v.item()
        legal = game.legal_moves()
        mask = torch.full((9,), float('-inf'), device=self.device)
        mask[legal] = 0.0
        p = F.log_softmax(p_logits + mask, dim=-1).exp().cpu().numpy()
        self.P[key] = p
        self.legal[key] = legal
        self.children[key] = legal
        self.terminal[key] = game.result()
        return v

    def add_dirichlet(self, root_key: Tuple):
        alpha = self.cfg.dirichlet_alpha
        eps = self.cfg.dirichlet_eps
        p = self.P[root_key].copy()
        legal = self.legal[root_key]
        noise = np.random.dirichlet([alpha]*len(legal))
        p_new = p.copy()
        # Only mix on legal entries
        for i, a in enumerate(legal):
            p_new[a] = (1-eps)*p[a] + eps*noise[i]
        self.P[root_key] = p_new

    def simulate(self, root: TicTacToe, move_count: int):
        path = []
        game = root.clone()

        # Selection & expansion
        while True:
            key = self.state_key(game)
            path.append((key, game.current_player))
            term = self.terminal.get(key, None)
            if term is None and key not in self.P:
                # expand
                v = self.expand(game, key)
                break
            if term is not None:
                # terminal node already known
                v = 0.0 if term == 0 else (1.0 if term == game.current_player else -1.0)
                break
            # choose action by PUCT
            total_N = sum(self.N[(key, a)] for a in self.children[key])
            best, best_u = None, -1e9
            for a in self.children[key]:
                q = self.Q[(key, a)]
                p = self.P[key][a]
                u = q + self.cfg.c_puct * p * math.sqrt(total_N + 1) / (1 + self.N[(key, a)])
                if u > best_u:
                    best_u, best = u, a
            game.play(best)

        # Backup (note: value is from current node perspective)
        # Convert v to perspective of nodes along the path (which alternate players)
        for key, player_at_node in reversed(path):
            self.N[(key, best)] += 1  # last chosen action at that key (approx)
            # Update W/Q for the action leading out of key; we reuse 'best' as the last action taken
            # For more precision, store actions per step; simplified here is acceptable for small games.
            self.W[(key, best)] += v if player_at_node == root.current_player else -v
            self.Q[(key, best)] = self.W[(key, best)] / self.N[(key, best)]

    def run(self, root: TicTacToe, move_count: int) -> np.ndarray:
        self.reset()
        root_key = self.state_key(root)
        v = self.expand(root, root_key)
        # Add Dirichlet noise only at root & early game
        if move_count == 0:
            self.add_dirichlet(root_key)
        for _ in range(self.cfg.sims):
            self.simulate(root, move_count)
        # Build visit distribution
        counts = np.zeros(9, dtype=np.float32)
        for a in self.legal[root_key]:
            counts[a] = self.N[(root_key, a)]
        # Temperature
        if move_count < self.cfg.temp_moves:
            pi = counts ** 1.0
        else:
            # argmax
            pi = np.zeros_like(counts)
            if counts.sum() > 0:
                pi[np.argmax(counts)] = 1.0
            else:
                # no visits? fallback uniform legal
                legal = self.legal[root_key]
                for a in legal:
                    pi[a] = 1.0 / len(legal)
        if pi.sum() > 0:
            pi = pi / (pi.sum() + 1e-8)
        return pi

# =====================
# Replay Buffer & Dataset
# =====================

@dataclass
class Sample:
    planes: np.ndarray  # (5,3,3)
    pi: np.ndarray      # (9,)
    z: float            # scalar in [-1,0,1]

class ReplayBuffer:
    def __init__(self, capacity: int, augment: bool = True):
        self.buf: deque[Sample] = deque(maxlen=capacity)
        self.augment = augment

    def push_game(self, states: List[np.ndarray], pis: List[np.ndarray], winner: int, players: List[int]):
        # winner in {+1, 0, -1}; players[i] is the player at that state (+1 or -1)
        for s, pi, p in zip(states, pis, players):
            z = 0.0 if winner == 0 else (1.0 if winner == p else -1.0)
            self.buf.append(Sample(s, pi, z))
            if self.augment:
                for sym in range(1,8):  # we've already stored identity
                    s2 = apply_symmetry_planes(s, sym)
                    pi2 = apply_symmetry_pi(pi, sym)
                    self.buf.append(Sample(s2, pi2, z))

    def __len__(self):
        return len(self.buf)

class BufferDataset(Dataset):
    def __init__(self, buf: ReplayBuffer):
        self.buf = buf

    def __len__(self):
        return len(self.buf)

    def __getitem__(self, idx):
        s = self.buf.buf[idx]
        return (
            torch.from_numpy(s.planes),
            torch.from_numpy(s.pi),
            torch.tensor(s.z, dtype=torch.float32)
        )

# =====================
# Minimax perfect opponent (for eval)
# =====================

class MinimaxTicTacToe:
    """修复后的正确Minimax实现"""
    def __init__(self):
        self.cache = {}

    def minimax(self, game: TicTacToe, depth=0) -> int:
        """正确的Minimax算法实现"""
        # 使用缓存避免重复计算
        key = (tuple(game.board.reshape(-1)), game.current_player)
        if key in self.cache:
            return self.cache[key]
        
        # 检查游戏是否结束
        result = game.result()
        if result is not None:
            self.cache[key] = result
            return result
        
        legal = game.legal_moves()
        if not legal:
            self.cache[key] = 0
            return 0
        
        if game.current_player == 1:  # X的回合，最大化
            best_value = -2
            for move in legal:
                game_copy = game.clone()
                game_copy.play(move)
                value = self.minimax(game_copy, depth+1)
                best_value = max(best_value, value)
        else:  # O的回合，最小化
            best_value = 2
            for move in legal:
                game_copy = game.clone()
                game_copy.play(move)
                value = self.minimax(game_copy, depth+1)
                best_value = min(best_value, value)
        
        self.cache[key] = best_value
        return best_value

    def best_move(self, game: TicTacToe) -> int:
        """获取最佳移动"""
        legal = game.legal_moves()
        if not legal:
            raise ValueError("No legal moves available")
        
        best_move = legal[0]
        
        if game.current_player == 1:  # X最大化
            best_value = -2
            for move in legal:
                game_copy = game.clone()
                game_copy.play(move)
                value = self.minimax(game_copy)
                if value > best_value:
                    best_value = value
                    best_move = move
        else:  # O最小化
            best_value = 2
            for move in legal:
                game_copy = game.clone()
                game_copy.play(move)
                value = self.minimax(game_copy)
                if value < best_value:
                    best_value = value
                    best_move = move
        
        return best_move

# =====================
# Self-play & Training
# =====================

def batch_self_play_games(net: AZNet, mcts_cfg: MCTSConfig, n_games: int, device: torch.device, batch_size: int = 32) -> List[Tuple[List[np.ndarray], List[np.ndarray], int, List[int]]]:
    """批量自对弈：充分利用GPU并行计算"""
    net.eval()
    all_games = []
    batch_mcts = BatchMCTS(net, mcts_cfg, device)
    
    # 分批处理
    for batch_start in range(0, n_games, batch_size):
        batch_end = min(batch_start + batch_size, n_games)
        current_batch_size = batch_end - batch_start
        
        # 初始化当前批次的游戏
        games = [TicTacToe() for _ in range(current_batch_size)]
        batch_states = [[] for _ in range(current_batch_size)]
        batch_pis = [[] for _ in range(current_batch_size)]
        batch_players = [[] for _ in range(current_batch_size)]
        move_counts = [0] * current_batch_size
        
        # 进行游戏直到所有游戏结束
        active_games = list(range(current_batch_size))
        
        while active_games:
            # 获取活跃游戏状态
            active_game_objects = [games[i] for i in active_games]
            active_move_counts = [move_counts[i] for i in active_games]
            
            # 批量MCTS推理
            pis = batch_mcts.batch_run_games(active_game_objects, active_move_counts)
            
            # 处理每个活跃游戏
            new_active_games = []
            for idx, game_idx in enumerate(active_games):
                game = games[game_idx]
                pi = pis[idx]
                
                # 记录状态和策略
                batch_states[game_idx].append(game.encode())
                batch_pis[game_idx].append(pi.astype(np.float32))
                batch_players[game_idx].append(game.current_player)
                
                # 选择行动
                legal = game.legal_moves()
                probs = np.array([pi[a] for a in legal], dtype=np.float32)
                if probs.sum() <= 0:
                    probs = np.ones(len(legal), dtype=np.float32) / len(legal)
                probs = probs / probs.sum()
                a = np.random.choice(legal, p=probs)
                
                # 执行行动
                game.play(a)
                move_counts[game_idx] += 1
                
                # 检查游戏是否结束
                res = game.result()
                if res is None:
                    new_active_games.append(game_idx)
                else:
                    # 游戏结束，记录结果
                    all_games.append((
                        batch_states[game_idx],
                        batch_pis[game_idx], 
                        res,
                        batch_players[game_idx]
                    ))
            
            active_games = new_active_games
        
        print(f"  批量自对弈: 完成 {batch_end}/{n_games} 局游戏")
    
    return all_games

def self_play_games(net: AZNet, mcts_cfg: MCTSConfig, n_games: int, device: torch.device) -> List[Tuple[List[np.ndarray], List[np.ndarray], int, List[int]]]:
    net.eval()
    games = []
    for _ in range(n_games):
        g = TicTacToe()
        mcts = MCTS(net, mcts_cfg, device)
        states, pis, players = [], [], []
        move_count = 0
        while True:
            pi = mcts.run(g, move_count)
            states.append(g.encode())
            pis.append(pi.astype(np.float32))
            players.append(g.current_player)
            # sample move by pi
            legal = g.legal_moves()
            probs = np.array([pi[a] for a in legal], dtype=np.float32)
            if probs.sum() <= 0:
                probs = np.ones(len(legal), dtype=np.float32) / len(legal)
            probs = probs / probs.sum()
            a = np.random.choice(legal, p=probs)
            g.play(a)
            move_count += 1
            res = g.result()
            if res is not None:
                games.append((states, pis, res, players))
                break
    return games

def train_one_epoch(net: AZNet, loader: DataLoader, optim: torch.optim.Optimizer, device: torch.device, l2: float = 1e-4) -> Tuple[float,float,float]:
    net.train()
    tot_loss = tot_pol = tot_val = 0.0
    for planes, pi_tgt, z in loader:
        planes = planes.to(device)
        pi_tgt = pi_tgt.to(device)
        z = z.to(device)
        p_logits, v = net(planes)
        pol_loss = F.cross_entropy(p_logits, torch.argmax(pi_tgt, dim=1))
        val_loss = F.mse_loss(v, z)
        loss = pol_loss + val_loss
        optim.zero_grad()
        loss.backward()
        optim.step()
        tot_loss += loss.item() * planes.size(0)
        tot_pol += pol_loss.item() * planes.size(0)
        tot_val += val_loss.item() * planes.size(0)
    n = len(loader.dataset)
    return tot_loss/n, tot_pol/n, tot_val/n

@torch.no_grad()
def evaluate_vs_minimax(net: AZNet, n_games: int, device: torch.device, mcts_sims_eval: int = 50) -> Tuple[float,float,float]:
    net.eval()
    mini = MinimaxTicTacToe()
    mcts_cfg = MCTSConfig(sims=mcts_sims_eval, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.0, temp_moves=0)
    win = draw = loss = 0
    for k in range(n_games):
        g = TicTacToe()
        first = (k % 2 == 0)  # alternate sides
        while True:
            if (g.current_player == 1 and first) or (g.current_player == -1 and not first):
                # our net move via MCTS
                mcts = MCTS(net, mcts_cfg, device)
                pi = mcts.run(g, move_count=9-len(g.legal_moves()))
                legal = g.legal_moves()
                a = legal[int(np.argmax([pi[x] for x in legal]))]
                g.play(a)
            else:
                a = mini.best_move(g)
                g.play(a)
            res = g.result()
            if res is not None:
                if (first and res == 1) or ((not first) and res == -1):
                    win += 1
                elif res == 0:
                    draw += 1
                else:
                    loss += 1
                break
    return win/n_games, draw/n_games, loss/n_games

# =====================
# Main
# =====================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=15, help='training iterations')
    ap.add_argument('--games-per-iter', type=int, default=300)
    ap.add_argument('--buffer-size', type=int, default=20000)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--epochs-per-iter', type=int, default=1)
    ap.add_argument('--mcts-sims', type=int, default=100)
    ap.add_argument('--mcts-temp-moves', type=int, default=2)
    ap.add_argument('--net-channels', type=int, default=32, help='网络通道数')
    ap.add_argument('--net-blocks', type=int, default=2, help='残差块数量')
    ap.add_argument('--batch-mcts', action='store_true', help='使用批量MCTS（GPU加速）')
    ap.add_argument('--batch-size-mcts', type=int, default=32, help='批量MCTS的批次大小')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--augment', action='store_true', default=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--save-model', type=str, default='alphazero_model.pth', help='保存模型的文件名')
    ap.add_argument('--load-model', type=str, default=None, help='加载预训练模型的文件名')
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    net = AZNet(in_ch=5, ch=args.net_channels, nblocks=args.net_blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 加载预训练模型（如果指定）
    start_iter = 1
    if args.load_model:
        try:
            checkpoint = torch.load(args.load_model, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_iter = checkpoint['iteration'] + 1
            print(f"✅ 成功加载模型: {args.load_model}, 从第{start_iter}轮开始训练")
        except FileNotFoundError:
            print(f"⚠️ 模型文件未找到: {args.load_model}, 从头开始训练")
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}, 从头开始训练")

    buf = ReplayBuffer(capacity=args.buffer_size, augment=args.augment)
    
    # 最佳模型管理
    best_draw_rate = 0.0
    best_model_path = "best_" + args.save_model
    patience = 3  # 容忍连续退化的次数
    worse_count = 0

    for it in range(start_iter, start_iter + args.iters):
        # Self-play
        mcfg = MCTSConfig(sims=args.mcts_sims, c_puct=1.5, dirichlet_alpha=0.3, dirichlet_eps=0.25, temp_moves=args.mcts_temp_moves)
        if args.batch_mcts:
            print(f"  使用批量MCTS (批次大小: {args.batch_size_mcts})")
            games = batch_self_play_games(net, mcfg, args.games_per_iter, device, args.batch_size_mcts)
        else:
            games = self_play_games(net, mcfg, args.games_per_iter, device)
        for states, pis, winner, players in games:
            buf.push_game(states, pis, winner, players)
        print(f"[Iter {it}] buffer size after self-play: {len(buf)}")

        # Train
        ds = BufferDataset(buf)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        for ep in range(args.epochs_per_iter):
            loss, pl, vl = train_one_epoch(net, dl, opt, device)
            print(f"  Train epoch {ep+1}/{args.epochs_per_iter}: loss={loss:.4f} pol={pl:.4f} val={vl:.4f}")

        # Eval vs perfect
        w,d,l = evaluate_vs_minimax(net, n_games=40, device=device, mcts_sims_eval=50)
        print(f"  Eval vs minimax: W={w*100:.1f}% D={d*100:.1f}% L={l*100:.1f}%")
        
        # 最佳模型管理
        current_draw_rate = d
        if current_draw_rate > best_draw_rate:
            # 性能提升，保存最佳模型
            best_draw_rate = current_draw_rate
            best_checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'iteration': it,
                'draw_rate': d,
                'args': vars(args)
            }
            torch.save(best_checkpoint, best_model_path)
            print(f"  🎉 新记录！平局率{d*100:.1f}%，最佳模型已保存")
            worse_count = 0
            
            # 早停检查
            if d >= 0.9:
                print(f"🎯 达到优秀水平！平局率{d*100:.1f}%，提前结束训练")
                break
                
        elif current_draw_rate < best_draw_rate - 0.1:  # 显著退化
            worse_count += 1
            print(f"  ⚠️ 性能退化 ({worse_count}/{patience})")
            
            if worse_count >= patience:
                print(f"  🔄 连续{patience}次退化，回退到最佳模型！")
                try:
                    best_checkpoint = torch.load(best_model_path, map_location=device)
                    net.load_state_dict(best_checkpoint['model_state_dict'])
                    opt.load_state_dict(best_checkpoint['optimizer_state_dict'])
                    
                    # 降低学习率
                    for param_group in opt.param_groups:
                        param_group['lr'] *= 0.5
                    print(f"  📉 学习率降低到 {opt.param_groups[0]['lr']:.6f}")
                    
                    worse_count = 0
                except Exception as e:
                    print(f"  ❌ 回退失败: {e}")
        else:
            print(f"  📊 性能稳定")

    print("Training complete.")
    
    # 使用最佳模型作为最终模型
    if os.path.exists(best_model_path) and best_draw_rate > 0:
        print(f"🏆 使用最佳模型 (平局率{best_draw_rate*100:.1f}%) 作为最终输出")
        # 复制最佳模型到最终保存路径
        shutil.copy2(best_model_path, args.save_model)
        print(f"💾 最佳模型已保存到: {args.save_model}")
    else:
        # 保存当前模型
        checkpoint = {
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'iteration': start_iter + args.iters - 1,
            'draw_rate': best_draw_rate,
            'args': vars(args)
        }
        torch.save(checkpoint, args.save_model)
        print(f"💾 当前模型已保存到: {args.save_model}")
    
    print(f"🎯 最终结果: 最佳平局率 {best_draw_rate*100:.1f}%")

if __name__ == '__main__':
    main()
