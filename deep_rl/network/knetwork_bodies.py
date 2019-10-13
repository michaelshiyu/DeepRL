from kernet.layers.kernelized_layer import kFullyConnected
from .network_utils import *

'''
class kFCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), sigmas=(10., 5.), gate=F.relu):
        super(kFCBody, self).__init__()
        dims = (state_dim,) + hidden_units
        # default: dims = (state_dim, 64, 64)
        self.layers = nn.ModuleList(
            [kFullyConnected(X=torch.rand(5, dim_in), n_out=dim_out, sigma=sigma, trainable_X=True) for dim_in, dim_out, sigma in zip(dims[:-1], dims[1:], sigmas)])
            # [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.gate = gate
        self.feature_dim = dims[-1]
        print(len(self.layers))

    def forward(self, x):
        for layer in self.layers:
            # x = self.gate(layer(x))
            print(layer.X)
            x = layer(x)
        return x
'''

class kFCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), sigma=1., gate=F.relu):
        super(kFCBody, self).__init__()
        self.layer0 = nn.Linear(state_dim, hidden_units[0])
        self.layer1 = kFullyConnected(X=torch.rand(5, hidden_units[0]), n_out=hidden_units[1], sigma=sigma, trainable_X=True)
        self.feature_dim = hidden_units[1]

    def forward(self, x):
        x = F.tanh(self.layer0(x))    
        x = self.layer1(x)
        return x


class kNatureConvBody(nn.Module):
    def __init__(self, in_channels=4, sigma=1.):
        super(NatureConvBody, self).__init__()
        self.feature_dim = 512
        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        # self.fc4 = layer_init(nn.Linear(7 * 7 * 64, self.feature_dim))
        self.fc4 = kFullyConnected(X=torch.rand(5, 7 * 7 * 64), n_out=self.feature_dim, sigma=sigma, trainable_X=True)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = F.relu(self.fc4(y))
        return y


