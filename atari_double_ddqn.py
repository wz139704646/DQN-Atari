from atari_ddqn import CnnDDQNAgent
import torch.nn.functional as F


class CnnDoubleDQNAgent(CnnDDQNAgent):
    """Double DQN with CNN"""

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
        # different from DQN
        # double DQN use model (not target model) to choose next actions
        next_a = self.model(s1).max(1)[1].view(-1, 1)
        # use the target network to estimate next Q (IMPORTANT: no grad.)
        q_values_next = self.target_model(s1).gather(1, next_a).detach()
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


# alias
double_dqn = CnnDoubleDQNAgent