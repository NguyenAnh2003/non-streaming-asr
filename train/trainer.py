from utils.utils import get_configs
from data_manipulation.dataloader import DevSet, TrainSet, TrainLoader, DevLoader
from torch.optim import Adam

params = get_configs("../configs/train_params.yaml")

print(f"Params: {params}")

# optimizer Adam

# loss function WER