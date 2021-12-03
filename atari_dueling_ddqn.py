from atari_ddqn import CnnDDQNAgent
from config import Config
from torch.optim import Adam
from buffer import RolloutStorage
from model import CnnDuelingDQN


class CnnDuelingDDQNAgent(CnnDDQNAgent):
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = RolloutStorage(config)
        self.model = CnnDuelingDQN(self.config.state_shape, self.config.action_dim)
        self.target_model = CnnDuelingDQN(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate,
                                eps=1e-5, weight_decay=0.95)

        if self.config.use_cuda:
            self.cuda()


# alias
dueling_dqn = CnnDuelingDDQNAgent
