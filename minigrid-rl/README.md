# Imitation Learning on Partially Observable MDPs
## Behavior Cloning
Train minigrid expert with
```bash
python -m scripts.train --algo ppo --env MiniGrid-Playground-v0 --model ppo_playground --save-interval 10 --frames 800000
```
Visualize expert and save trajectory with
```bash
python -m scripts.visualize --env MiniGrid-Playground-v0 --model ppo_playground
```
or add --render False for fast trajectory gathering (a lot faster!)
```bash
python3 -m scripts.visualize --env MiniGrid-Playground-v0 --model ppo_playground --render False
```
hardcode expert
```bash
python -m scripts.visualize --env MiniGrid-MemoryS13-v0 --save_dir MemoryS13-train --hardcode --episodes 500

Train BC with
```bash
python minigrid-rl/scripts/bc_train.py --env MiniGrid-Playground-v0 --data_dir expert_data/empty
```

Evaluate BC with
```bash
python minigrid-rl/scripts/bc_eval.py --env MiniGrid-Empty-16x16-v0 --model "BC checkpoint path"
```

## DP
```bash
python -m scripts.visualize --env MiniGrid-DoorKey-8x8-v0 --dp
python -m scripts.visualize --env MiniGrid-MemoryS7-v0 --dp --rounds 3
```
## BC with Memory
train
```bash
python scripts/bc_train.py --env MiniGrid-MemoryS7-v0 --train_data_dir expert_data/memory_train --val_data_dir expert_data/memory_val --name memory_stack --epochs 1000 --seed 42 --encoding stack --history_size 5
```
eval
```bash
python scripts/bc_eval.py --env MiniGrid-MemoryS7-v0 --model memory_5_best.pth --history_size 5 --render --seed 43
```
## LSTM with Memory
train
```shell
python -m scripts.bc_train --env MiniGrid-MemoryS7-v0 --train_data_dir expert_data/memory_train --val_data_dir expert_data/memory_val --name memory_lstm --epochs 100 --seed 42 --encoding lstm
```
eval
```shell
python scripts/lstm_eval.py --env MiniGrid-MemoryS7-v0 --model memory_lstm_best.pth --render --seed 43
```

