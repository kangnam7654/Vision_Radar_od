import json
import os

import cv2

from utils.common.project_paths import GetPaths
from utils.inference.inference_modules import InferenceModules
import yaml
from box import Box
from tqdm import tqdm

def config_load(config_yaml='mapsea_config.yaml'):
    with open(GetPaths().get_yaml_folder(config_yaml)) as f:
        config = yaml.full_load(f)
        config = Box(config)
    return config


class InferenceProcess(InferenceModules):
    """
    Inference 에 대한 프로세스를 작성한 클래스입니다.
    """
    def __init__(self, cfg):
        """

        :param data_path: Inference 를 진행할 데이터가 있는 경로입니다.
        :param circle_info: 레이더 내 원의 좌표에 대한 변수입니다.
        """
        super().__init__()
        self.paths = GetPaths()
        self.cfg = cfg
        self.data_list, self.circle_information, self.model = self.load()

    def load(self):
        """ 데이터와 레이더 내 원의 좌표, 그리고 학습된 모델의 가중치를 불러오는 메서드입니다.

        :param data_path: Inference 를 진행할 데이터가 있는 경로입니다.
        :param circle_info: 레이더 내 원의 좌표에 대한 변수입니다.
        :param weight_path: 학습된 weight의 경로입니다.
        :return:
        data_list: 불러온 데이터의 list 입니다.
        circle_information: 데이터 내의 원 좌표입니다.
        model: 데이터를 inference 할 모델입니다.
        """
        data_list = sorted(self.get_data_list(path=self.cfg.inference.data_dir))
        circle_information = self.load_circle_coordinate_dict(self.cfg.inference.circle_info_dir)
        weight = self.load_weight(weight=self.cfg.inference.weight)
        model = self.load_model(weight=weight)
        return data_list, circle_information, model

    def process(self):
        """ Inference 를 진행하는 프로세스부 입니다.

        :param save: Inference 후, 이미지와 라벨정보의 저장 여부를 결정하는 변수입니다.
        :param save_dir: save 의 이미지와 라벨정보 저장을 할 폴더의 입력입니다.
        :param show: inference중에 이미지의 표시 여부입니다.
        """
        if self.cfg.inference.save:
            save_dir_ = os.path.join(self.paths.project_root, self.cfg.inference.save_dir)
            os.makedirs(save_dir_, exist_ok=True)
            print(f'결과가 "{save_dir_}" 에 저장됩니다.')

        for data in tqdm(self.data_list, desc='inference'):
            data_file_name = os.path.split(data)[1]
            original_image, image_to_inference, output = self.inference(
                self.model, data, self.circle_information
            )
            xyxyn = self.ratio_to_coordinate(output)  # 라벨들의 xy 좌표의 비율, class, class_name을 가지는 dict로 반환
            original_xyxyn = self.crop_to_original(xyxyn, self.circle_information)  # 자른 원형의 좌표를 자르기 전 좌표로 변환
            labeled_image = self.display(original_image, original_xyxyn, show=self.cfg.inference.save)

            if self.cfg.inference.save:
                image_path = os.path.join(save_dir_, data_file_name)
                label_path = f"{image_path.split('.')[0]}.json"
                cv2.imwrite(image_path, labeled_image)
                with open(label_path, "w") as f:
                    json.dump(original_xyxyn, f)

            # print(f"{idx + 1}/{len(self.data_list)} completed")


def main():
    config = config_load()
    mapsea = InferenceProcess(cfg=config)
    mapsea.process()

if __name__ == "__main__":
    main()