from dataset import MyDataset
from torch.utils.data import DataLoader

class Loader:
    def __init__(self, args):
        self.train_loader = None
        if args.train:
            self.train_loader = DataLoader(
                MyDataset(args, 'train'),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_threads,
                pin_memory=not args.cpu,
                drop_last=True
            )
        self.test_loader = DataLoader(
            MyDataset(args, 'test'),
            batch_size=1,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=not args.cpu
        )