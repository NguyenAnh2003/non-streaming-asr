from pytorch_lightning import Trainer
from models.sim_clr import SimCLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
  model = SimCLR()
  trainer = Trainer()