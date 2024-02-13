from torch.utils.data import DataLoader, Dataset

""" define custom dataset """
class TrainSet(Dataset):
    def __init__(self):
        """ define init """

    def __getitem__(self, item):
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample

        # transform audio to mel spec

        return

class DevSet(Dataset):
    def __init__(self):
        """ define init """
    def __getitem__(self, item):
        """ return log mel spectrogram, and transcript """

        # load audio to array and sample

        # transform audio to mel spec

        return

""" define data loader """
# custom dataloader
class TrainLoader(DataLoader):
    def __init__(self):
        """ Train loader init """

class DevLoader(DataLoader):
    def __init__(self):
        """ Dev loader init """

""" check """
if __name__ == "__main__":
    pass