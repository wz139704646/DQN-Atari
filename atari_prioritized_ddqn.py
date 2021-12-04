from atari_ddqn import CnnDDQNAgent
from config import Config
from torch.optim import Adam
from model import CnnDQN
from buffer import PrioritizedRolloutStorage
import torch.nn.functional as F


class CnnPrioritizedDDQNAgent(CnnDDQNAgent):
    """DQN with prioritized experience replay buffer"""

    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = PrioritizedRolloutStorage(config)
        self.model = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model = CnnDQN(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate,
                                eps=1e-5, weight_decay=0.95)

        if self.config.use_cuda:
            self.cuda()

    def learning(self, fr):
        s0, s1, a, r, done, indices = self.buffer.sample(self.config.batch_size)
        # importance sampling weights
        weights = self.buffer.get_weight(indices)
        if self.config.use_cuda:
            s0 = s0.float().to(self.config.device)/255.0
            s1 = s1.float().to(self.config.device)/255.0
            a = a.to(self.config.device)
            r = r.to(self.config.device)
            done = done.to(self.config.device)
            weights = weights.to(self.config.device)

        # calculate TD loss
        q_values = self.model(s0).gather(1, a)
        # use the target network to estimate next Q
        q_values_next = self.target_model(s1).max(1)[0].view(-1, 1)
        # target q values
        q_values_tar = r + self.config.gamma * q_values_next * (1-done)
        td_err = q_values_tar - q_values
        # mse loss with importance sampling weight
        loss = (td_err.pow(2) * weights).mean()

        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()

        # update buffer weights
        self.buffer.update_weight(indices, td_err)
        # update beta
        beta = self.config.buff_beta + (self.config.buff_beta_final - self.config.buff_beta) \
            * (fr / self.config.beta_anneal_steps)
        self.buffer.update_beta(beta)

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()


# alias
prio_dqn = CnnPrioritizedDDQNAgent