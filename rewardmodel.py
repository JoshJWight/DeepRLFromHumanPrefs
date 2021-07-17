import pickle
import random
import collections
import numpy as np
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
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.value(conv_out)


class RewardModel:
    def __init__(self, input_shape, n_actions, device, load=False):
        self.memoryFile = "rewardmemory.dat"
        self.modelFile = "rewardmodel.dat"

        self.device = device
        self.net = RewardNet(input_shape, n_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, eps=1e-3)

        self.memoryLimit = 3000
        self.memory = collections.deque(maxlen=self.memoryLimit)

        self.clipLimit = 100
        self.clipStorage = collections.deque(maxlen=self.clipLimit)

        self.nTrainSteps = 0
        self.nPreTrain = 10000
    
        print(f"Load: {load}")
        if load:
            self.net.load_state_dict(torch.load(self.modelFile))
            self.net.eval()
            self.memory=pickle.load(open(self.memoryFile, "rb"))
            self.nTrainSteps = nPreTrain + 1
            print("Loaded reward model")
            print(f"Memory size: {len(self.memory)}")

    def save(self):
        torch.save(self.net.state_dict(), self.modelFile)
        pickle.dump(self.memory, open(self.memoryFile, "wb"))
    
    def storeComparison(self, clip1, clip2, p1, p2):
        self.memory.append((clip1, clip2, p1, p2))

    def storeClip(self, clip):
        self.clipStorage.append(clip)


    def newComparison(self):
        #TODO the paper recommends having an ensemble of reward models and having the human judge comparisons where they don't agree
        return random.sample(self.clipStorage, 2)
        
        

    def evaluate(self, state):
        #TODO the paper recommends regularization since the scale of the reward model is arbitrary.
        return self.net(state).detach()

    def train(self, n_samples):
        if len(self.memory) < n_samples:
            print(f"Reward model not ready to train - only has {len(self.memory)} comparisons")
            return

        samples = random.sample(self.memory, n_samples)

        losses = []
        for (clip1, clip2, p1, p2) in samples:

            #Might be able to do this in a single pass with a higher-dimensional tensor

            #assume clips are arrays of states
            clip1_v = torch.FloatTensor(np.array(clip1, copy=False)).to(self.device)
            clip2_v = torch.FloatTensor(np.array(clip2, copy=False)).to(self.device)

            self.optimizer.zero_grad()

            expsum1 = torch.exp(self.net(clip1_v).sum())
            expsum2 = torch.exp(self.net(clip2_v).sum())

            phat1 = expsum1 / (expsum1 + expsum2)
            phat2 = expsum2 / (expsum2 + expsum1)

            loss = -1.0 * ((p1 * torch.log(phat1)) + (p2 * torch.log(phat2)))
            losses.append(loss)
            
            if torch.isnan(loss).any():
                print("NaN Alert!")
                print(f"expsums: {expsum1}, {expsum2} phats: {phat1}, {phat2} loss: {loss}")

            #TODO track stuff

        totalLoss = sum(losses)
        totalLoss.backward()
        self.optimizer.step()

        self.nTrainSteps += n_samples
        print(f"Reward model loss: {totalLoss}")
        if not self.isReady():
            print(f"Total steps: {self.nTrainSteps}/{self.nPreTrain}")
             

    def isReady(self):
        return self.nTrainSteps > self.nPreTrain
