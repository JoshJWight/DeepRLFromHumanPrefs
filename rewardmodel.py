
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

#Note that this design doesn't care about the action the agent took, only the state it reached.
#This means it can't really give an opinion on actions that end the episode.
class RewardNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RewardNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.value(conv_out)

def trainRewardNet(net, optimizer, clip1, clip2,  p1, p2)
    

class RewardModel:
    def __init__(self, input_shape, n_actions):
        self.net = RewardNet(input_shape, n_actions)
        self.optimizer = optim.Adam(net.parameters(), lr=0.001, eps=1e-3)
        self.memory = collections.deque()
        
        self.memoryLimit = 3000
        
    def remember(clip1, clip2, p1, p2):
        if len(self.memory) >= self.memoryLimit:
            self.memory.popleft()
        self.memory.append((clip1, clip2, p1, p2))

    def evaluate(state):
        #TODO the paper recommends regularization since the scale of the reward model is arbitrary.
        return self.net(state).detach()

    def train(n_samples):
        samples = random.sample(self.memory, n_samples)
        for (clip1, clip2, p1, p2) in samples:
            #this only trains on one clip pair at a time. might be better to train on multiple clip pairs at once

            #assume clips are arrays of states
            clip1_v = torch.FloatTensor(np.array(clip1, copy=False)).to(device)
            clip2_v = torch.FloatTensor(np.array(clip2, copy=False)).to(device)

            self.optimizer.zero_grad()

            expsum1 = torch.exp(self.net(clip1_v).sum())
            expsum2 = torch.exp(self.net(clip2_v).sum())

            phat1 = expsum1 / (expsum1 + expsum2)
            phat2 = expsum2 / (expsum2 + expsum1)

            loss = -1.0 * ((p1 * torch.log(phat1)) + (p2 * torch.log(phat2)))

            loss.backward()

            self.optimizer.step()

            #TODO track stuff
             
