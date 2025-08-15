import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ===== 数据读取 =====
df = pd.read_csv("double_chromosphere.csv")
reds = df[['red1','red2','red3','red4','red5','red6']].values
blues = df[['blue']].values

# ===== one-hot 编码 =====
def one_hot_encode(numbers, num_classes):
    enc = np.zeros(num_classes)
    for n in numbers:
        enc[n-1] = 1  # 号码从1开始
    return enc

X_data = []
y_red_data = []
y_blue_data = []

window_size = 10  # 用最近10期预测下一期
for i in range(len(df) - window_size):
    seq_red = []
    seq_blue = []
    for j in range(window_size):
        seq_red.append(one_hot_encode(reds[i+j], 33))   # 红球 one-hot
        seq_blue.append(one_hot_encode(blues[i+j], 16)) # 蓝球 one-hot
    X_data.append(np.hstack([np.array(seq_red).sum(axis=0)/6,  # 红球频率特征
                             np.array(seq_blue).sum(axis=0)])) # 蓝球频率特征
    
    y_red_data.append(one_hot_encode(reds[i+window_size], 33))
    y_blue_data.append(one_hot_encode(blues[i+window_size], 16))

X = torch.tensor(X_data, dtype=torch.float32)
y_red = torch.tensor(y_red_data, dtype=torch.float32)
y_blue = torch.tensor(y_blue_data, dtype=torch.float32)

# ===== 模型定义 =====
class DoubleBallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(DoubleBallLSTM, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc_red = nn.Linear(hidden_size, 33)   # 红球输出
        self.fc_blue = nn.Linear(hidden_size, 16)  # 蓝球输出
    
    def forward(self, x):
        h = self.relu(self.fc1(x))
        red_out = torch.softmax(self.fc_red(h), dim=1)
        blue_out = torch.softmax(self.fc_blue(h), dim=1)
        return red_out, blue_out

model = DoubleBallLSTM(input_size=X.shape[1])
criterion = nn.BCELoss()  # 多标签分类损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ===== 训练 =====
epochs = 200
for epoch in range(epochs):
    optimizer.zero_grad()
    red_pred, blue_pred = model(X)
    loss_red = criterion(red_pred, y_red)
    loss_blue = criterion(blue_pred, y_blue)
    loss = loss_red + loss_blue
    loss.backward()
    optimizer.step()
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.6f}")

# ===== 预测下一期 =====
last_seq_red = one_hot_encode(reds[-window_size:], 33)
last_seq_blue = one_hot_encode(blues[-window_size:], 16)

# 用最近 window_size 期的频率特征作为输入
X_last = np.hstack([np.array([one_hot_encode(r, 33) for r in reds[-window_size:]]).sum(axis=0)/6,
                    np.array([one_hot_encode(b, 16) for b in blues[-window_size:]]).sum(axis=0)])
X_last = torch.tensor(X_last, dtype=torch.float32).unsqueeze(0)

red_pred, blue_pred = model(X_last)

# Top-N 推荐
top_reds = torch.topk(red_pred[0], 6).indices.numpy() + 1
top_blues = torch.topk(blue_pred[0], 1).indices.numpy() + 1

print("预测红球候选:", sorted(top_reds))
print("预测蓝球候选:", top_blues[0])
