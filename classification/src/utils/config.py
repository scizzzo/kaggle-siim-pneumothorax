import json

TRAIN_DIR = 'train_dir'
DATA_PREFIX = 'data_prefix'
FOLD_ID = 'fold_id'

INPUT_SIZE = 'input_size'
BATCH_SIZE = 'batch_size'
EPOCHS = 'epochs'
LR = 'lr'

MODEL_NAME = 'model_name'
MODEL_PARAMS = 'model_params'
LOSS_NAME = 'loss_name'
LOSS_PARAMS = 'loss_params'
OPTIM_NAME = 'optim_name'
OPTIM_PARAMS = 'optim_params'
SCHEDULER_NAME = 'scheduler_name'
SCHEDULER_PARAMS = 'scheduler_params'

PARAMS_POSTFIX = '_params'


class Config:
    def __init__(self, data):
        self._data = data

    def __getitem__(self, item):
        if PARAMS_POSTFIX in item:
            return self._data.get(item, {})
        else:
            return self._data[item]

    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
            json_data = json.load(f)
        return cls(json_data)

    def save_json(self, dst_path):
        with open(dst_path) as f:
            json.dump(self._data, f, indent=4)

    def to_json(self):
        return json.dumps(self._data, indent=4)