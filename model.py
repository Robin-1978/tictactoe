import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #  shortcut连接，如果输入输出通道数不同，则使用1x1卷积调整
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=3, input_channels=64, res_channels=[32, 64], dropout_rate=0.3):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.input_channels = input_channels  # 第一层卷积的通道数
        self.res_channels = res_channels     # 残差块的通道数数组
        self.dropout_rate = dropout_rate
        
        # 输入层：3个通道（棋盘状态 + 当前玩家标记 + 常数层）→ input_channels
        self.input_conv = nn.Conv2d(3, input_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm2d(input_channels)
        
        # 残差块：使用res_channels数组配置
        self.res_blocks = nn.ModuleList()
        in_channels = input_channels  # 第一个残差块的输入来自input_conv
        for out_channels in res_channels:
            self.res_blocks.append(ResidualBlock(in_channels, out_channels))
            in_channels = out_channels  # 下一个块的输入通道数
        
        # 最后一个残差块的输出通道数
        final_channels = res_channels[-1] if res_channels else input_channels
        
        # 策略头 - 参考成功实现的设计
        policy_channels = max(final_channels // 4, 8)
        self.policy_conv = nn.Conv2d(final_channels, policy_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_dropout = nn.Dropout(dropout_rate)
        self.policy_fc = nn.Linear(policy_channels * board_size * board_size, board_size * board_size)

        # 价值头 - 参考成功实现的设计
        value_channels = max(final_channels // 4, 8)
        self.value_conv = nn.Conv2d(final_channels, value_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        
        # 简单的2层结构，适合井字棋
        value_input_size = value_channels * board_size * board_size
        value_hidden_size = max(final_channels // 2, 32)
        self.value_dropout = nn.Dropout(dropout_rate)
        self.value_fc1 = nn.Linear(value_input_size, value_hidden_size)
        self.value_fc2 = nn.Linear(value_hidden_size, 1)

        # 智能设备选择：根据网络规模和硬件特性选择最优设备
        self.device = self._select_optimal_device()
        self.to(self.device)
    
    def _select_optimal_device(self):
        """智能选择最优计算设备"""
        if not torch.cuda.is_available():
            return torch.device("cpu")
        
        # 计算网络参数总数
        total_params = sum(p.numel() for p in self.parameters())
        
        # 小网络(<1M参数)在CPU上可能更高效，避免GPU开销
        if total_params < 1_000_000:
            # 对于井字棋这种小网络，测试GPU vs CPU性能
            try:
                # 快速基准测试
                import time
                test_input = torch.randn(1, 3, 3, 3)
                
                # 测试CPU
                self.to('cpu')
                test_input_cpu = test_input.to('cpu')
                start = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        _ = self.forward(test_input_cpu)
                cpu_time = time.time() - start
                
                # 测试GPU
                self.to('cuda')
                test_input_gpu = test_input.to('cuda')
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        _ = self.forward(test_input_gpu)
                    torch.cuda.synchronize()
                gpu_time = time.time() - start
                
                # 选择更快的设备，并添加安全边际
                if cpu_time * 1.2 < gpu_time:  # CPU需要明显更快才选择
                    selected_device = torch.device("cpu")
                    reason = f"小网络CPU更优(CPU:{cpu_time:.4f}s vs GPU:{gpu_time:.4f}s)"
                else:
                    selected_device = torch.device("cuda")
                    reason = f"GPU仍有优势(CPU:{cpu_time:.4f}s vs GPU:{gpu_time:.4f}s)"
                    
                print(f"🎯 智能设备选择: {selected_device} - {reason}")
                return selected_device
                
            except Exception as e:
                print(f"⚠️ 设备基准测试失败，使用CPU: {e}")
                return torch.device("cpu")
        else:
            # 大网络优先使用GPU
            print(f"🚀 大网络({total_params:,}参数)使用GPU加速")
            return torch.device("cuda")

    def forward(self, state):
        # 输入层
        x = F.relu(self.input_bn(self.input_conv(state)))
        
        # 通过残差块
        for block in self.res_blocks:
            x = block(x)
        
        # 策略头 - 加入dropout防止过拟合
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, self.policy_conv.out_channels * self.board_size * self.board_size)
        policy = self.policy_dropout(policy)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # 价值头 - 加入dropout防止过拟合
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.value_conv.out_channels * self.board_size * self.board_size)
        value = self.value_dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy, value = self.forward(state)
        return policy.cpu().numpy(), value.cpu().numpy()