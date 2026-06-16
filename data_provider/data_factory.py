from torch.utils.data import DataLoader

from data_provider.data_loader import UEAloader
from data_provider.uea import collate_fn


def data_provider(args, flag):
    Data = UEAloader
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = args.batch_size

    data_set = Data(
        args=args,
        root_path=args.root_path,
        flag=flag,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    return data_set, data_loader
