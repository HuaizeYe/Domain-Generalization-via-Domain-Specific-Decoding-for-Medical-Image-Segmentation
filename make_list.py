import os

if __name__ == '__main__':
    base_dir = '/data/ziqi/datasets/muti_site_med/Fundus'
    domains = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
    for domain in domains:
        train_list_file = open(os.path.join(base_dir, domain, 'train.list'), 'w')
        test_list_file = open(os.path.join(base_dir, domain, 'test.list'), 'w')
        # train_list_file = open(domain + '_train.list', 'w')
        # test_list_file = open(domain + '_test.list', 'w')

        train_list = os.listdir(os.path.join(base_dir, domain, 'train', 'ROIs', 'image'))
        test_list = os.listdir(os.path.join(base_dir, domain, 'test', 'ROIs', 'image'))
        
        for train_ids in train_list:
            # train_list_file.write(domain + '/train/ROIs/image/' + train_ids + ' ' + domain + '/train/ROIs/mask/' + train_ids + '\n')
            train_list_file.write('train/ROIs/image/' + train_ids + ' ' + 'train/ROIs/mask/' + train_ids + '\n')
            train_list_file.flush()
        train_list_file.close()
        
        for test_ids in test_list:
            # test_list_file.write(domain + '/test/ROIs/image/' + test_ids + ' ' + domain + '/test/ROIs/mask/' + test_ids + '\n')
            test_list_file.write('test/ROIs/image/' + test_ids + ' ' + 'test/ROIs/mask/' + test_ids + '\n')
            test_list_file.flush()
        test_list_file.close()