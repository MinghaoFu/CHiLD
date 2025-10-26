from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_Human, Dataset_Humaneva, Dataset_CESM2, \
    Dataset_fmri, Dataset_MuJoco, Dataset_Weather, Dataset_WeatherBench
from torch.utils.data import DataLoader

data_dict = {
    'CESM2': Dataset_CESM2,
    'ETTh1': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'Human': Dataset_Human,
    'Humaneva': Dataset_Humaneva,
    'fmri': Dataset_fmri,
    'MuJoco': Dataset_MuJoco,
    'weather': Dataset_Weather,
    'WeatherBench': Dataset_WeatherBench,
}


def data_provider(args, flag, draw):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    shuffle_flag = False if flag == 'test' or draw == 1 else True
    drop_last = False if flag == 'test' or draw == 1 else True
    batch_size = args.batch_size
    freq = args.freq
    data_set = Data(
        args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False)
    return data_set, data_loader
