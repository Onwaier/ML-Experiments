import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out)


if __name__ == '__main__':

    net = LinearRegression()
    inputs = torch.randn((128, 4))
    outputs = net(inputs)
    print(net)
    print(outputs)