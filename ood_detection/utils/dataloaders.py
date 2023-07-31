from torch.utils.data import DataLoader


class InfiniteDataIterator:
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library
    A data iterator that will never stop producing data
    """

    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            print("Reached the end, resetting data loader...")
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)
