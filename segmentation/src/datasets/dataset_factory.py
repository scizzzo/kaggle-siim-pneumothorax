from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size, sampler=None):
    is_train = 'train' == dataset.stage

    dataloader = DataLoader(dataset,
                            shuffle=is_train if sampler is None else False,
                            batch_size=batch_size,
                            drop_last=is_train,
                            sampler=sampler if is_train else None,
                            num_workers=8,
                            pin_memory=False)
    return dataloader
