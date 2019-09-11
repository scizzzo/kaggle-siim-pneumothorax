import os


def prepare_train_directories(out_dir):
    os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)
