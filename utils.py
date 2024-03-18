import gzip
import pickle
import yaml


# 加载Phonexi-2014T数据集
def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return dict(loaded_object)


def config():
    with open('./config.yaml', 'r') as file:
        conf = yaml.safe_load(file)
    return conf


config = config()
