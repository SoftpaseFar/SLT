from utils import load_dataset_file
from utils import config

train_label_path = config['data']['train_label_path']

print(load_dataset_file(train_label_path)['train/27January_2013_Sunday_tagesschau-8842'])
