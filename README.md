# DQN trained on Atari

DQN and some variants (Double DQN, Dueling DQN, DQN with Prioritized Experience Replay Buffer) trained on Atari environments.

## usage

```
$python main.py --help

usage: main.py [-h] [--algo ALGO] [--train] [--env ENV] [--test] [--retrain] [--model_path MODEL_PATH] [--no-cuda] [--cuda_id CUDA_ID]

optional arguments:
  -h, --help            show this help message and exit
  --algo ALGO           the algorithm to use
  --train               train model
  --env ENV             gym environment
  --test                test model
  --retrain             retrain model
  --model_path MODEL_PATH
                        if test or retrain, import the model
  --no-cuda             disables CUDA training
  --cuda_id CUDA_ID     if test or retrain, import the model
```

## files

Core files:

- `atari_ddqn.py`: Implementation of DQN algorithm.
- `atari_double_ddqn.py`: Implementation of Double DQN algorithm.
- `atari_dueling_ddqn.py`: Implementation of Dueling DQN algorithm.
- `atari_prioritized_ddqn.py`: Implementation of DQN with Prioritized Experience Replay Buffer.
- `buffer.py`: Experience replay buffer (including prioritized version).
- `model.py`: Q networks (including dueling version).
