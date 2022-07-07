import copy
import glob
import json
import os

import cv2
import torch

from utils.common.project_paths import GetPaths


class InferenceModules:
    """
    Inference 프로세스를 작성하기 위해 필요한 모듈들을 모아놓은 클래스입니다.
    """
    def __init__(self):
        self.paths = GetPaths()
        self.root_dir = GetPaths().project_root
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_circle_coordinate_dict(self, circle_info_path):
        """ 원 좌표의 정보를 담은 json 을 불러오는 메서드입니다.
        :param file: json 파일을 입력으로 받습니다.
        :return:
        """
        file = self.paths.get_project_root(circle_info_path)
        with open(self.paths.get_circle_coordinate_folder(file), "r") as f:
            circle_coordinate_dict = json.load(f)
        return circle_coordinate_dict

    def load_weight(self, weight='default'):
        """ 모델의 가중치를 불러오는 메서드입니다. default 는 gray 입니다.

        :param weight:
        :return:
        """
        # if weight == "default" or "gray" or None:
        #     weight_path = os.path.join(
        #         self.root_dir, "runs", "train", "circle_gray", "weights", "best.pt"
        #         # self.root_dir, 'exp15', 'weights', 'best.pt'
        #     )
        # elif weight == "color" or "colour":
        #     weight_path = os.path.join(
        #         self.root_dir, "runs", "train", "circle", "weights", "best.pt"
        #     )
        # else:
        weight_path = self.paths.get_project_root(weight)
        return weight_path

    def load_model(self, weight=None):
        """ 모델을 불러오는 메서드 입니다.

        :param weight: 모델의 가중치를 입력하는 변수입니다.
        :return:
        """
        path_model_folder = os.path.join(self.root_dir)
        if os.path.isfile(weight):
            path_weight = weight
        else:
            path_weight = "yolov5m.pt"

        model = torch.hub.load(
            path_model_folder,
            "custom",
            path=path_weight,
            source="local",
            force_reload=True,
            verbose=True,
        )
        model = model.to(self.device)
        return model

    def get_data_list(self, path, extension="png"):
        """ inference 할 데이터의 리스트를 불러오는 메서드입니다.
        :param path: 데이터의 경로입니다.
        :param extension: 데이터의 확장자입니다. 기본값은 png 입니다.
        :return:
        """
        data_folder = self.paths.get_project_root(path)
        data_image_list = glob.glob(
            os.path.join(data_folder, "**", f"*.{extension}"), recursive=True
        )
        return data_image_list

    @staticmethod
    def inference_preprocess(image, circle_information):
        """ inference 할 이미지를 전처리 하는 메서드입니다.

        원 좌표를 받으면 해당 좌표에 따라 이미지를 추출하고 자릅니다. 이후 자른 이미지를 반환합니다.

        :param image: 이미지 파일을 입력 받습니다.
        :param circle_information: 이미지의 원 좌표를 입력 받습니다.
        :return:
        """
        image_to_inference = copy.deepcopy(image)

        circle_centre_x = circle_information['old_radar']["x_centre"]
        circle_centre_y = circle_information['old_radar']["y_centre"]
        circle_r = int(circle_information['old_radar']["radius"])

        for i in range(500):
            cv2.circle(
                image_to_inference,
                center=(circle_centre_x, circle_centre_y),
                radius=circle_r + (i * 2),
                color=(0, 0, 0),
                thickness=2,
            )
        cut_x_start = circle_centre_x - circle_r
        cut_x_end = circle_centre_x + circle_r
        cut_y_start = circle_centre_y - circle_r
        cut_y_end = circle_centre_y + circle_r
        image_to_inference = image_to_inference[
            cut_y_start:cut_y_end, cut_x_start:cut_x_end
        ]
        return image_to_inference

    def save_pre_inference_image(self, imgs_pre):
        path_saved = glob.glob(os.path.join(self.root_dir, "runs", "hub", "*"))
        latest_folder = None
        if len(path_saved) != 1:
            latest_num = 0
            for i in path_saved[1:]:
                result = i.split("exp")
                num = int(result[1])
                if latest_num <= num:
                    latest_num = num
                latest_folder = f"{result[0]}exp{latest_num}"
        else:
            latest_folder = path_saved[0]

        for idx, pre_image in enumerate(imgs_pre):
            number = format(idx, "04")
            f = f"image{number}_pre.png"
            cv2.imwrite(os.path.join(latest_folder, f), pre_image)

    def inference(self, model, image_file, circle_information, mode="gray"):
        """ inference 를 시행하는 메서드입니다.

        :param model:
        :param image_file:
        :param circle_information:
        :param mode:
        :return:
        """
        # mode_lower = mode.lower()
        # if mode_lower == "gray":
        #     src = cv2.imread(image_file, 0)
        #     src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        # elif mode_lower == "color" or "colour" or "bgr":
        #     src = cv2.imread(image_file)
        # else:
        #     raise Exception('mode should be "gray" or "color" or "colour" or "BGR"')

        src_colour = cv2.imread(image_file)
        src_gray = cv2.imread(image_file, 0)
        src = cv2.cvtColor(src_gray, cv2.COLOR_GRAY2BGR)

        image_to_inference = copy.deepcopy(src)
        image_to_inference = self.inference_preprocess(
            image_to_inference, circle_information
        )
        output = model(image_to_inference)
        return src_colour, image_to_inference, output

    @staticmethod
    def display(image, dic, show=False):
        """ inference 한 이미지를 label 좌표를 받아 이미지에 표시하는 메서드입니다.

        :param image:
        :param dic:
        :param show:
        :return:
        """
        for value in dic.values():
            pt1 = (value["xmin"], value["ymin"])
            pt2 = (value["xmax"], value["ymax"])
            text_pt = (value["xmax"] + 5, value["ymin"])
            cv2.rectangle(image, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(
                image,
                text=f"{value['name']}: {round(value['confidence'], 2)}",
                org=text_pt,
                fontFace=0,
                fontScale=0.7,
                color=(255, 255, 255),
                thickness=2,
            )
        if show:
            cv2.imshow("a", image)
            cv2.waitKey(1)
        return image

    @staticmethod
    def ratio_to_coordinate(output):
        """ inference 의 결과에서 라벨의 위치 비율을 받아 좌표로 변환하는 메서드입니다.

        :param output:
        :return:
        """
        shape = output.imgs[0].shape[:2]
        original_x, original_y = shape[0], shape[1]
        xyxyn = output.pandas().xyxyn[0].T.to_dict()
        for i in xyxyn.keys():
            xyxyn[i]["xmin"] = int(xyxyn[i]["xmin"] * original_x)
            xyxyn[i]["xmax"] = int(xyxyn[i]["xmax"] * original_x)
            xyxyn[i]["ymin"] = int(xyxyn[i]["ymin"] * original_y)
            xyxyn[i]["ymax"] = int(xyxyn[i]["ymax"] * original_y)
        return xyxyn

    @staticmethod
    def crop_to_original(xyxyn, circle_information):
        """ 자른 원형의 표를 자르기 전 원본의 좌표로 변환하는 메서드입니다.

        :param xyxyn:
        :param circle_information:
        :return:
        """
        x_centre = circle_information['old_radar']["x_centre"]
        y_centre = circle_information['old_radar']["y_centre"]
        radius = circle_information['old_radar']["radius"]
        adjust_x_start = x_centre - radius
        adjust_y_start = y_centre - radius

        original_xyxyn = copy.deepcopy(xyxyn)
        for key, value in original_xyxyn.items():
            original_xyxyn[key] = value
            original_xyxyn[key]["xmin"] = original_xyxyn[key]["xmin"] + adjust_x_start
            original_xyxyn[key]["xmax"] = original_xyxyn[key]["xmax"] + adjust_x_start
            original_xyxyn[key]["ymin"] = original_xyxyn[key]["ymin"] + adjust_y_start
            original_xyxyn[key]["ymax"] = original_xyxyn[key]["ymax"] + adjust_y_start
        return original_xyxyn


if __name__ == "__main__":
    pass
