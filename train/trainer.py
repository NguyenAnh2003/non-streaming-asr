from utils.utils import get_configs
from data.dataloader import DevSet, TrainSet, TrainLoader, DevLoader

params = get_configs("../configs/train_params.yaml")

print(f"Params: {params}")