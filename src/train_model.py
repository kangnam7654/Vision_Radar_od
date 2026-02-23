import os
from ultralytics import YOLO
from utils.common.project_paths import GetPaths
from box import Box

import yaml


def config_load(config_yaml='radar_config.yaml'):
    with open(GetPaths().get_yaml_folder(config_yaml), 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)
    config = Box(config)
    return config


def main():
    config = config_load()
    conf = config.train_conf
    hyp = config.hyp

    model = YOLO(conf.model)
    model.train(
        data=conf.dataset_yaml,
        epochs=conf.epochs,
        batch=conf.batch_size,
        imgsz=conf.image_size,
        device=conf.device,
        patience=conf.patience,
        **hyp,
    )


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
