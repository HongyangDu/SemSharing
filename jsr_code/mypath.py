class Path(object):
    @staticmethod
    def dataset_root_dir(dataset):
        if dataset == 'cityscapes' or dataset =="cityscapes_2class":
            return '/mnt/datagrid/public_datasets/CityScapes/'     # folder that contains leftImg8bit/
        if dataset == 'LaF':
            return '/mnt/datagrid/public_datasets/lost_and_found/'     # folder that contains leftImg8bit/
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
