run everything under skill_based/, such that packages will work.
always use direct import such as :
from agents.models import LSTMPolicy, TransformerPolicy


### train agent
```shell
python -m scripts.robust_train --env MiniGrid-MemoryS13-v0 --epochs 500 --train_data_dir expert_data/train_data --val_data_dir  expert_data/val_data/
```