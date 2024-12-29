### Run under skill_based with ```-m``` tag.


### test agent on env
* BC
```shell
python -m scripts.agent_eval --env MiniGrid-MemoryS11-v0 --map_size 11 --agent bc --encoder lstm --episodes 100 --max_steps 30 --render
```
use ```--agent fake``` to generate random traj.

* Explore agent\
we still need to pass in policy_ckpt_path because we use its LSTM to encode our traj.
```shell
python -m scripts.agent_eval --env MiniGrid-MemoryS11-v0 --map_size 11 --agent explore --model "exp name in storage/"
```
### online RL fine-tune
```shell
python -m scripts.finetune --env MiniGrid-MemoryS7-v0 --ckpt_path memory_stack/best.pth --name ppo-bc
```
PPO without finetune
```shell
python -m scripts.finetune --env MiniGrid-MemoryS7-v0 --name ppo
```
### train explore agent
```shell
python -m scripts.train_agent.train_explore --algo PPO --env MiniGrid-MemoryS7-v0 --disc_path ckpt/discriminator/model.pt --recurrence 4
```
### train discriminator
The latent dimension of LSTM is 256
Discriminator takes in 256 dim embedding and outputs a score.
```shell
python -m scripts.train_agent.train_discriminator --bc_traj traj_data/bc_traj.npy --fake_traj traj_data/fake_traj.npy
```

### train explore and discriminator adversarially
* we have to throw in a initial fake random traj to kickstart the discriminator\
* disc_epochs is the epochs for discriminator training, lower it to prevent discriminator from overpowering.
```bash
python -m scripts.train_agent.train_adversarial --env MiniGrid-MemoryS11-v0 --map_size 11 --exp_name 11_first_time --episodes 1000 --disc_epochs 5 --GAIL_iters 10 
```

### Change this line to move the agent to middle in memory.py
```python
        # Fix the player's start position and orientation
        # self.agent_pos = np.array((4, height // 2))
        # self.agent_dir = 3
        self.agent_pos = np.array((1, height // 2))
        self.agent_dir = 0
```
