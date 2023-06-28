from dataloaders.datasets import cityscapes, cityscapes_2class, lostandfound 
from torch.utils.data import DataLoader
from mypath import Path

def make_data_loader(cfg, **kwargs):
    if cfg.DATASET.TRAIN == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(cfg, root=Path.dataset_root_dir(cfg.DATASET.TRAIN), split='train')
    elif cfg.DATASET.TRAIN == 'cityscapes_2class':
        train_set = cityscapes_2class.CityscapesSegmentation_2Class(cfg, root=Path.dataset_root_dir(cfg.DATASET.TRAIN), split='train')
    else:
        raise NotImplementedError

    if cfg.DATASET.VAL == 'cityscapes':
        val_set = cityscapes.CityscapesSegmentation(cfg, root=Path.dataset_root_dir(cfg.DATASET.VAL), split='val')
    elif cfg.DATASET.VAL == 'cityscapes_2class':
        val_set = cityscapes_2class.CityscapesSegmentation_2Class(cfg, root=Path.dataset_root_dir(cfg.DATASET.VAL), split='val')
    elif cfg.DATASET.VAL == 'LaF':
        val_set = lostandfound.LostAndFound(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='val')
    else:
        raise NotImplementedError

    if cfg.DATASET.TEST == 'cityscapes':
        test_set = cityscapes.CityscapesSegmentation(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='test')
    elif cfg.DATASET.TEST == 'cityscapes_2class':
        test_set = cityscapes_2class.CityscapesSegmentation_2Class(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='test')
    elif cfg.DATASET.TEST == 'LaF':
        test_set = lostandfound.LostAndFound(cfg, root=Path.dataset_root_dir(cfg.DATASET.TEST), split='test')
    else:
        raise NotImplementedError


    train_loader = DataLoader(train_set, batch_size=cfg.INPUT.BATCH_SIZE_TRAIN, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg.INPUT.BATCH_SIZE_TEST, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=cfg.INPUT.BATCH_SIZE_TEST, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, train_set.NUM_CLASSES
