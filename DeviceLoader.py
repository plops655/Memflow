class DeviceLoader:

    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for batch in self.dataloader:
            yield [inp.to(self.device, non_blocking=True) for inp in batch]

    def __len__(self):
        return len(self.dataloader)