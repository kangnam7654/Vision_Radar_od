import os
from ultralytics import YOLO
from utils.common.project_paths import GetPaths
from utils.train.pre_process_modules import PreProcessModules
from box import Box

import yaml


class RadarTrain(PreProcessModules):
    def __init__(self, cfg):
        """
        :param cfg: radar_config.yaml
        """
        super().__init__()
        self.cfg = cfg
        self.raw_data_dir1 = self.cfg.preprocess_conf.raw_data_dir1
        self.raw_data_dir2 = self.cfg.preprocess_conf.raw_data_dir2
        self.train_files1 = self.files_to_train_list(self.raw_data_dir1)
        self.train_files2 = self.files_to_train_list(self.raw_data_dir2)
        self.all_train_files = [self.train_files1, self.train_files2]

    def pre_process(self):
        config = self.cfg.preprocess_conf.copy()  # configurations

        # legacy
        if config.original_data_save:
            classes = self.make_classes(self.train_files1, circle=False, update_yaml=False)
            self.labelme_to_yolo(self.train_files1,
                                 classes,
                                 save=True,
                                 save_dir=config.original_save_dir,
                                 circle=False)

        # circle
        classes_circle = self.make_classes(self.all_train_files, circle=True, update_yaml=config.update_yaml)
        self.labelme_to_yolo(self.all_train_files,
                             classes_circle,
                             save=True,
                             save_dir=config.circle_data_save_dir,
                             circle=True,
                             circle_info_save_path=config.circle_info_save_path,
                             number_to_count=config.circle_n_count
        )
        circle_dir = self.paths.get_project_root(config.circle_data_save_dir)
        self.folder_split(circle_dir)  # 폴더 나누기(train, valid, test[empty])
        self.data_split(circle_dir, ratio=config.split_ratio)  # 데이터 나누기 (train to valid)

        if config.also_gray:
            gray_dir = self.paths.get_project_root(config.gray_save_dir)
            os.makedirs(gray_dir, exist_ok=True)
            self.copy_to_gray(circle_dir, gray_dir)

        if config.update_yaml:
            with open(self.paths.get_project_root('yaml', 'radar_dataset.yaml')) as f:
                d_yaml = yaml.full_load(f)
            d_yaml['path'] = gray_dir
            with open(self.paths.get_project_root('yaml', 'radar_dataset.yaml'), 'w') as f:
                yaml.dump(d_yaml, f)
        del config

    def model_train(self):
        conf = self.cfg.train_conf.copy()
        model = YOLO(conf.model)
        model.train(
            data=conf.dataset_yaml,
            epochs=conf.epochs,
            batch=conf.batch_size,
            imgsz=conf.image_size,
            device=conf.device,
        )


def config_load(config_yaml='radar_config.yaml'):
    with open(GetPaths().get_yaml_folder(config_yaml), 'r', encoding='utf-8') as f:
        config = yaml.full_load(f)
    config = Box(config)
    return config


def train_with_pre_process():
    """ 데이터세트 전처리가 되지 않았을 경우의 train입니다.

    :return:
    """
    config = config_load()
    pipeline = RadarTrain(cfg=config)
    pipeline.pre_process()
    pipeline.model_train()


def train_without_pre_process():
    """ 데이터세트 전처리가 끝났을경우의 train입니다.

    :return:
    """
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


def main(run_pre_process=False):
    if run_pre_process:
        train_with_pre_process()
    else:
        train_without_pre_process()


if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main(run_pre_process=False)
