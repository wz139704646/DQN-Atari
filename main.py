import torch
import argparse
from config import Config
from tester import Tester
from trainer import Trainer
from atari_ddqn import *
from atari_double_ddqn import *
from atari_dueling_ddqn import *
from atari_prioritized_ddqn import *
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--algo', type=str, default='dqn', help='the algorithm to use')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument('--cuda_id', type=str, default='0', help='if test or retrain, import the model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.05
    config.eps_decay = 30000
    config.frames = 2000000
    config.use_cuda = args.cuda
    config.learning_rate = 1e-4
    config.init_buff = 10000
    config.max_buff = 80000
    config.learning_interval = 4
    config.update_tar_interval = 1000
    config.batch_size = 32
    config.gif_interval = 20000
    config.print_interval = 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 500000
    config.win_reward = 18
    config.win_break = True
    config.device = torch.device("cuda:"+args.cuda_id if args.cuda else "cpu")
    config.output = os.path.join(config.output, args.algo)
    # for prioritized experience replay buffer
    config.buff_alpha = 0.5
    config.buff_beta = 0.4
    config.buff_beta_final = 1.
    config.beta_anneal_steps = 1000000

    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    print(config.action_dim, config.state_shape)
    agent = globals()[args.algo](config)

    if args.train:
        trainer = Trainer(agent, env, config)
        trainer.train()

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test()

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)

        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        trainer.train(fr)
