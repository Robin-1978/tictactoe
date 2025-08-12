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
    def __init__(self, board_size=3, num_res_blocks=2, channels=[16, 32, 32]):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        
        # 输入层
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels[1])


        # 添加残差块
        # self.res_blocks = nn.ModuleList()
        # for i in range(num_res_blocks):
        #     in_ch = channels[i]
        #     out_ch = channels[i+1]
        #     self.res_blocks.append(ResidualBlock(in_ch, out_ch))

        # 策略头
        # 动态计算策略头通道数：channels[-1]的1/8，最小为8
        policy_channels = max(channels[-1] // 4, 8)
        self.policy_conv = nn.Conv2d(channels[-1], policy_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * board_size * board_size, board_size * board_size)

        # 价值头
        # 动态计算价值头通道数：channels[-1]的1/16，最小为4
        value_channels = max(channels[-1] // 8, 4)
        self.value_conv = nn.Conv2d(channels[-1], value_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * board_size * board_size, 32)
        self.value_fc2 = nn.Linear(32, 1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        # for block in self.res_blocks:
        #     x = block(x)
        x = F.relu(self.bn2(self.conv2(x)))
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, self.policy_conv.out_channels * self.board_size * self.board_size)
        policy = F.log_softmax(self.policy_fc(policy), dim=1)

        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.value_conv.out_channels * self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

    def predict(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            policy, value = self.forward(state)
        return policy.cpu().numpy(), value.cpu().numpy()