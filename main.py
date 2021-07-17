#Adapted from github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

import random
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common
import picker
import rewardmodel

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
#To be clear, this has nothing to do with the video clips. This was here in the original code
CLIP_GRAD = 0.1

#But this wasn't
CLIP_SIZE=25
CLIPS_PER_BATCH=5


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

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
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, rewardModel, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    not_done_idx = []
    last_states = []
    for idx, hist in enumerate(batch):
        states.append(np.array(hist[0].state, copy=False))
        actions.append(int(hist[0].action))

        if len(hist) > REWARD_STEPS:
            last_states.append(np.array(hist[REWARD_STEPS].state, copy=False))
            not_done_idx.append(idx)

    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    rewards_v = rewardModel.evaluate(states_v)
    actions_t = torch.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = rewards_v.data.cpu().numpy().flatten()
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    for hist in random.sample(batch, CLIPS_PER_BATCH):
        #TODO this can include clips of different lengths. Could interfere with training nets on many clips at once
        clip = []
        for exp in hist:
            clip.append(np.array(exp.state))
        rewardModel.storeClip(clip)

    return states_v, actions_t, ref_vals_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    parser.add_argument("-l", "--load", default=False, action="store_true", help="Load previous progress from files")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    if not args.load:
        print("Are you sure you want to start without loading? Any saved progress will be overwritten. (y/n)")
        if not input() == "y":
            print("Loading. Remember to set the command line flag -l next time.")
            args.load = True
        else:
            print("OK, continuing without loading.")


    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(f"Env shape: {envs[0].observation_space.shape}")
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSource(envs, agent, steps_count=CLIP_SIZE)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)


    rewardModel = rewardmodel.RewardModel(envs[0].observation_space.shape, envs[0].action_space.n, device, load=args.load)
    pickerWindow = picker.PickerWindow()
    pickerWindow.show_all()
    pickerWindow.gtkMain() 

    batch = []

    batchnum=0

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):

                envs[0].render()                

                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue
                
                batchnum +=1

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, rewardModel,  device=device)
                batch.clear()

                rewardModel.train(30) 
                

                if rewardModel.isReady():
                    #TODO move this whole block into a separate file
                    optimizer.zero_grad()
                    logits_v, value_v = net(states_v)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                    log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                    loss_policy_v = -log_prob_actions_v.mean()

                    prob_v = F.softmax(logits_v, dim=1)
                    entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                    # calculate policy gradients only
                    loss_policy_v.backward(retain_graph=True)
                    grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                            for p in net.parameters()
                                            if p.grad is not None])

                    # apply entropy and value gradients
                    loss_v = entropy_loss_v + loss_value_v
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()
                    # get full loss
                    loss_v += loss_policy_v

                    tb_tracker.track("advantage",       adv_v, step_idx)
                    tb_tracker.track("values",          value_v, step_idx)
                    tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                    tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                    tb_tracker.track("loss_value",      loss_value_v, step_idx)
                    tb_tracker.track("loss_total",      loss_v, step_idx)
                    tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                    tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                    tb_tracker.track("grad_var",        np.var(grads), step_idx)

                #Handle clip comparisons and reward model training
                if not pickerWindow.judgingClip:
                    if pickerWindow.hasResult:
                        clips, result = pickerWindow.getResult()
                        if result is not picker.PickerResult.DISCARD:
                            p1 = 0
                            p2 = 0
                            if result is picker.PickerResult.LEFT:
                                p1 = 1
                            elif result is picker.PickerResult.RIGHT:
                                p2 = 1
                            elif result is picker.PickerResult.SAME:
                                p1 = 0.5
                                p2 = 0.5
                            rewardModel.storeComparison(clips[0], clips[1], p1, p2)
                    pickerWindow.setClips(rewardModel.newComparison())

                if batchnum % 10 == 0:
                    rewardModel.save()
                    print("saved")







