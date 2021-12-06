import os
import random
import torch
from torch.optim import Adam
import torch.nn.functional as F
from buffer import RolloutStorage
from config import Config
from core.util import get_class_attr_val
from model import CnnDQN
import numpy as np


class CnnDDQNAgent:
    """DQN with CNN"""
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = RolloutStorage(config)
        self.model = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        # self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate,
        #                         eps=1e-5, weight_decay=0.95)
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()

    def act(self, state, epsilon=None):
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float)/255.0
            if self.config.use_cuda:
                state = state.to(self.config.device)
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr):
        s0, s1, a, r, done, _ = self.buffer.sample(self.config.batch_size)
        if self.config.use_cuda:
            s0 = s0.float().to(self.config.device)/255.0
            s1 = s1.float().to(self.config.device)/255.0
            a = a.to(self.config.device)
            r = r.to(self.config.device)
            done = done.to(self.config.device)

        # calculate TD loss
        q_values = self.model(s0).gather(1, a)
        # use the target network to estimate next Q (IMPORTANT: no grad.)
        q_values_next = self.target_model(s1).detach().max(1)[0].view(-1, 1)
        # target q values
        q_values_tar = r + self.config.gamma * q_values_next * (1-done)
        # mse loss
        loss = F.mse_loss(q_values, q_values_tar)

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def cuda(self):
        self.model.to(self.config.device)
        self.target_model.to(self.config.device)

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)

    def save_model(self, output, name=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr


# alias
dqn = CnnDDQNAgent