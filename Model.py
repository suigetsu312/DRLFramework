import torch.nn as nn

class CNNLayer(nn.Module):
    def __init__(self, in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 padding,
                 activation):
        super(CNNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation
    def forward(self, x):
        return self.activation(self.conv(x))
class MLP(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super(MLP, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = activation
    def forward(self, x):
        x = self.fc(x)
        x = self.activation(x)
        return x
