# by CDHZAYN

import torch
import torch.nn as nn
import torchvision


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 7)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(7, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def train_and_test(iteration=10):
    # 定义预处理管道，将图像转为Tensor后进行标准化
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # 训练阶段
    MNIST_train_set = torchvision.datasets.MNIST("mnist_dataset/", train=True, download=True, transform=transform)
    # 将MNIST_train_set转变为测试数据集X与真实标签集y
    X = MNIST_train_set.data.reshape(-1, 28 * 28).float()
    y = MNIST_train_set.targets

    # 创建MLP模型实例
    model = MLP()
    loss = nn.CrossEntropyLoss()
    # 添加adam优化器
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(iteration):
        # 优化器梯度清零
        optimizer.zero_grad()
        output = model(X)
        result_loss = loss(output, y)  # 根据输出和目标计算出损失值
        result_loss.backward()  # 求出神经网络当中的参数的梯度，以方便之后对神经网络参数进行的更新
        # 使用优化器进行更新
        optimizer.step()

    weights = {}

    # 测试阶段
    with torch.no_grad():
        model.eval()  # 将模型设置为评估模式，这会影响一些层（例如Dropout）的行为

        # 加载测试数据集
        MNIST_test_set = torchvision.datasets.MNIST("mnist_dataset/", train=False, download=True, transform=transform)
        X_test = MNIST_test_set.data.reshape(-1, 28 * 28).float()
        y_test = MNIST_test_set.targets

        # 预测测试集数据
        output_test = model(X_test)
        _, predicted_labels = torch.max(output_test, dim=1)

        # 计算正确率
        accuracy = (predicted_labels == y_test).sum().item() / len(y_test)
        print("Accuracy: {:.2f}%".format(accuracy * 100))

        # 可视化权重
        # 获取每层的权重矩阵
        weights['fc1'] = model.fc1.weight.data
        weights['fc2'] = model.fc2.weight.data

    return weights
