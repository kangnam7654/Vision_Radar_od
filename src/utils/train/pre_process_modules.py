import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))
DATA_DIR = ROOT_DIR.joinpath('data')

import glob
import json
import shutil
import cv2
import numpy as np

from utils.common.project_paths import GetPaths
from tqdm import tqdm
import random
import yaml


class PreProcessModules:
    def __init__(self):
        self.paths = GetPaths
        self.except_class = ['range', 'object', 'terra']

    @staticmethod
    def files_to_train_list(directory, *args, image_extension="png"):
        image_list = sorted(
            glob.glob(
                os.path.join(directory, *args, "**", f"*.{image_extension}"),
                recursive=True,
            )
        )
        json_list = sorted(
            glob.glob(os.path.join(directory, *args, "**", "*.json"), recursive=True)
        )
        t = Path(image_list[0])
        no_suffix_images = [Path(image_file).with_suffix('') for image_file in image_list]  # 확장자 없는 이미지 리스트
        no_suffix_jsons = [Path(json_file).with_suffix('') for json_file in json_list]  # 확장자 없는 json 리스트

        # IMAGE 파일과 LABEL 파일 둘 다 존재하는 경우에만 데이터로 씀
        file_set = set(no_suffix_images) & set(no_suffix_jsons)
        files_to_train = sorted(list(file_set))
        return files_to_train

    @staticmethod
    def filter_outlier(values_list) -> list:
        """ 1 시그마 이내의 데이터를 제외한 나머지 필터링
        :param values_list:
        :return:
        """
        sigma_under = np.mean(values_list) - np.std(values_list)
        sigma_over = np.mean(values_list) + np.std(values_list)
        not_outlier_list = [
            value for value in values_list if sigma_under <= value <= sigma_over
        ]
        return not_outlier_list

    def filter_circle_information(self, files_to_train, number_to_count=10):
        circle_x_list, circle_y_list, circle_r_list = [], [], []
        for file in tqdm(files_to_train[:number_to_count], desc='원형좌표계산'):
            if Path(file).suffix == "png": # suffix 존해할 경우
                image_file = file
            else:
                image_file = Path(file).with_suffix('.png') # suffix 존재 하지 않을 경우

            circle_x_centre, circle_y_centre, circle_r = self.get_circle_information(
                image_file
            )
            circle_x_list.append(circle_x_centre)
            circle_y_list.append(circle_y_centre)
            circle_r_list.append(circle_r)

        # 아웃라이어 필터링 및 평균 계산
        circle_x_centre = np.round(np.mean(self.filter_outlier(circle_x_list)))
        circle_y_centre = np.round(np.mean(self.filter_outlier(circle_y_list)))
        circle_r = np.round(np.mean(self.filter_outlier(circle_r_list)))
        return int(circle_x_centre), int(circle_y_centre), int(circle_r)

    def make_classes(self, file_list_no_extension, circle=True, update_yaml=True):
        dic_classes = {}
        labeled_class_list = []
        for files in file_list_no_extension:
            for file in files:  # 확장자 없는 파일 리스트
                json_file = f"{file}.json"
                with open(json_file, "r") as f:
                    labelme_json = json.load(f)  # load labelme json

                shapes = labelme_json["shapes"]
                if circle:  # 원형일 경우
                    for i in range(len(shapes)):
                        labeled_class = shapes[i]["label"].replace(" ", "_")  # 클래스명
                        if labeled_class == 'boat':
                            labeled_class = 'boat_triangle'
                        if labeled_class not in self.except_class + labeled_class_list:
                            labeled_class_list.append(labeled_class)

                # remain this for legacy
                else:
                    for i in range(len(shapes)):
                        labeled_class = (
                            shapes[i]["label"].replace(" ", "_").replace("ragne", "range")
                        )
                        if labeled_class not in labeled_class_list:
                            labeled_class_list.append(labeled_class)

        for idx, name in enumerate(sorted(labeled_class_list)):
            dic_classes[name] = idx

        if update_yaml:
            with open(self.paths.get_project_root('yaml', 'mapsea_dataset.yaml')) as f:
                d_yaml = yaml.full_load(f)

            d_yaml['nc'] = len(labeled_class_list)
            d_yaml['names'] = sorted(labeled_class_list)

            with open(self.paths.get_project_root('yaml', 'mapsea_dataset.yaml'), 'w') as f_:
                yaml.dump(d_yaml, f_)
        return dic_classes

    @staticmethod
    def get_circle_information(source_file_path):
        src = cv2.imread(str(source_file_path), 0)  # gray scale load

        circles = cv2.HoughCircles(
            src,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=20,
            param2=10,
            minRadius=0,
            maxRadius=0,
        )  # 원 추출

        circles = np.uint16(np.around(circles))
        circle = circles[0][0]

        circle_centre_x = int(circle[0])
        circle_centre_y = int(circle[1])
        circle_r = int(circle[2])
        return circle_centre_x, circle_centre_y, circle_r

    @staticmethod
    def folder_split(folder):
        for k in ["train", "valid", "test"]:
            make_folder = os.path.join(folder, k)
            os.makedirs(make_folder, exist_ok=True)

    @staticmethod
    def copy_to_gray(source_folder, destination_folder, exist_ok=True):
        shutil.copytree(source_folder, destination_folder, dirs_exist_ok=exist_ok)
        image_list = sorted(glob.glob(os.path.join(destination_folder, "*", "*.png")))

        for i in tqdm(image_list, desc='grayscale 생성'):
            image = cv2.imread(i, 0)
            cv2.imwrite(i, image)

    @staticmethod
    def data_split(folder, ratio=0.8):
        image_list = sorted(glob.glob(os.path.join(folder, "*.png")))
        txt_list = sorted(glob.glob(os.path.join(folder, "*.txt")))

        new_image = []
        new_text = []
        old_image = []
        old_text = []

        for i, t in zip(image_list, txt_list):
            if 'new' in i:
                new_image.append(i)
                new_text.append(t)
            else:
                old_image.append(i)
                old_text.append(t)

        dst_train = os.path.join(folder, "train")
        dst_valid = os.path.join(folder, "valid")

        for image_file, text_file in tqdm(zip(new_image, new_text), desc='데이터 분할'):

            image_file_name = os.path.split(image_file)[1]
            text_file_name = os.path.split(text_file)[1]

            select_folder = random.choices(
                [dst_train, dst_valid], weights=[ratio, 1 - ratio], k=1
            )


            shutil.move(image_file, os.path.join(select_folder[0], image_file_name))
            shutil.move(text_file, os.path.join(select_folder[0], text_file_name))

        for image_file, text_file in tqdm(zip(old_image, old_text), desc='데이터 분할'):

            image_file_name = os.path.split(image_file)[1]
            text_file_name = os.path.split(text_file)[1]

            # select_folder = random.choices(
            #     [dst_train, dst_valid], weights=[ratio, 1 - ratio], k=1
            # )
            shutil.copy(image_file, os.path.join(dst_train, image_file_name))
            shutil.copy(text_file, os.path.join(dst_train, text_file_name))

            shutil.move(image_file, os.path.join(dst_valid, image_file_name))
            shutil.move(text_file, os.path.join(dst_valid, text_file_name))

    def labelme_to_yolo(
            self,
            file_list_no_extension,
            dic_classes,
            save=False,
            save_dir=None,
            circle=False,
            circle_info_save_path=None,
            number_to_count=10,
    ):
        dic_labels = {}
        file_name_map = {}
        num = 0

        if circle:  # 원형 좌표 구해서 저장
            circle_dic = {}
            for files in file_list_no_extension:
                p = Path(files[0]).parts
                if 'new_radar' in p:
                    key = 'new_radar'
                else:
                    key = 'old_radar'
                circle_x_centre, circle_y_centre, circle_r = self.filter_circle_information(
                    files_to_train=files,
                    number_to_count=number_to_count,
                )
                r_adjust = 10
                circle_r = circle_r + r_adjust
                draw_centre = (int(circle_x_centre), int(circle_y_centre))
                circle_coordinate = {
                    "x_centre": draw_centre[0],
                    "y_centre": draw_centre[1],
                    "radius": circle_r,
                }
                circle_dic[key] = circle_coordinate
            circle_info_save_dir, _ = os.path.split(circle_info_save_path)
            os.makedirs(circle_info_save_dir, exist_ok=True)
            with open(
                self.paths.get_project_root(circle_info_save_path),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(circle_dic, f)

        for files in tqdm(file_list_no_extension, desc='좌표변환'):
            for file in files:
                image_file = f"{file}.png"
                json_file = f"{file}.json"

                if circle:
                    p = Path(file)
                    if 'new_radar' in p.parts:
                        t = 'new_radar'  # type
                    else:
                        t = 'old_radar'
                    info = circle_dic[t]
                    circle_x_centre = info['x_centre']
                    circle_y_centre = info['y_centre']
                    circle_r = info['radius']
                    draw_centre = (int(circle_x_centre), int(circle_y_centre))
                    image_to_circle = cv2.imread(image_file)
                    radius_amp = 0
                    for _ in range(300):
                        cv2.circle(
                            img=image_to_circle,
                            center=draw_centre,
                            radius=int(circle_r) + radius_amp,
                            color=(0, 0, 0),
                            thickness=2,
                        )
                        radius_amp += 2

                    cut_x_start = int(circle_x_centre - circle_r)
                    cut_x_end = int(circle_x_centre + circle_r)
                    cut_y_start = int(circle_y_centre - circle_r)
                    cut_y_end = int(circle_y_centre + circle_r)
                    image_to_circle = image_to_circle[
                        cut_y_start:cut_y_end, cut_x_start:cut_x_end, :
                    ]
                else:
                    t = None

                new_file_name = f"{t}_{str(num).zfill(6)}"  # 저장 파일 명
                file_name_map[new_file_name] = {}
                file_name_map[new_file_name]["name"] = file

                dic_labels[new_file_name] = {}

                with open(json_file, "r") as f:
                    labelme_json = json.load(f)
                shapes = labelme_json["shapes"]
                image_width = labelme_json["imageWidth"]
                image_height = labelme_json["imageHeight"]

                for idx, shape in enumerate(shapes):
                    fixed_label = shape["label"].replace(" ", "_").replace("ragne", "range") # replace -> '_' 대신 ' '로 오타 낸 것들을 수정
                    if fixed_label == 'boat':
                        fixed_label = 'boat_triangle'
                    try:
                        label = dic_classes[fixed_label]
                    except:
                        continue

                    start_point, end_point = shape["points"]
                    x_start, y_start = start_point
                    x_end, y_end = end_point

                    if circle and shape['label'] not in ['range', 'ragne']:  # filtering
                        x_start = x_start - cut_x_start
                        x_end = x_end - cut_x_start
                        y_start = y_start - cut_y_start
                        y_end = y_end - cut_y_start

                        image_width = image_to_circle.shape[0]
                        image_height = image_to_circle.shape[1]

                    x_centre = (x_start + x_end) / 2
                    x_centre_normalize = round(x_centre / image_width, 6)

                    y_centre = (y_start + y_end) / 2
                    y_centre_normalize = round(y_centre / image_height, 6)

                    width = abs((x_end - x_start))
                    width_normalize = round(width / image_width, 6)

                    height = abs((y_end - y_start))
                    height_normalize = round(height / image_height, 6)

                    # if label != 3:
                    box = f"{label} {x_centre_normalize} {y_centre_normalize} {width_normalize} {height_normalize}"
                    dic_labels[new_file_name][idx] = box
                # print(f"[{num + 1}/{len(file_list_no_extension)}] 번째 변환 끝")

                if save:
                    if circle:
                        image_to_save = image_to_circle
                    else:
                        image_to_save = cv2.imread(image_file)
                    save_dir_ = self.paths.get_project_root(save_dir)
                    os.makedirs(save_dir_, exist_ok=True)
                    image_to_save_file_name = os.path.join(save_dir_, f"{new_file_name}.png")
                    label_to_save_file_name = os.path.join(save_dir_, f"{new_file_name}.txt")
                    cv2.imwrite(image_to_save_file_name, image_to_save)

                    with open(label_to_save_file_name, "w") as txt:
                        for label in dic_labels[new_file_name].values():
                            txt.write(f"{label}\n")

                num += 1
        return dic_labels, file_name_map


# 레거시
def labelme_to_yolo_legacy(
        dir,
        dic_classes=None,
        save=False,
        save_folder=None,

):
    dic_labels = {}
    file_name_map = {}
    num = 0

    directory = Path(dir)
    image_files = directory.glob('*.png')
    json_files = directory.glob('*.json')

    for image_file in tqdm(image_files, desc='좌표변환'):
        json_file = image_file.with_suffix('.json')

        new_file_name = f"{str(num).zfill(6)}"
        file_name_map[new_file_name] = {}
        file_name_map[new_file_name]["name"] = file

        dic_labels[new_file_name] = {}

        with open(json_file, "r") as f:
            labelme_json = json.load(f)
        shapes = labelme_json["shapes"]
        image_width = labelme_json["imageWidth"]
        image_height = labelme_json["imageHeight"]

        for idx, shape in enumerate(shapes):
            label = dic_classes[
                shape["label"].replace(" ", "_").replace("ragne", "range")
            ]  # replace -> '_' 대신 ' '로 오타 낸 것들을 수정

            start_point, end_point = shape["points"]
            x_start, y_start = start_point
            x_end, y_end = end_point

            x_centre = (x_start + x_end) / 2
            x_centre_normalize = round(x_centre / image_width, 6)

            y_centre = (y_start + y_end) / 2
            y_centre_normalize = round(y_centre / image_height, 6)

            width = abs((x_end - x_start))
            width_normalize = round(width / image_width, 6)

            height = abs((y_end - y_start))
            height_normalize = round(height / image_height, 6)

            # if label != 3:
            box = f"{label} {x_centre_normalize} {y_centre_normalize} {width_normalize} {height_normalize}"
            dic_labels[new_file_name][idx] = box
            # print(f"[{num + 1}/{len(file_list_no_extension)}] 번째 변환 끝")

        if save:
            image_to_save = cv2.imread(image_file)
            save_dir = self.paths.get_project_root(*save_folder)
            os.makedirs(save_dir, exist_ok=True)
            image_to_save_file_name = os.path.join(save_dir, f"{new_file_name}.png")
            label_to_save_file_name = os.path.join(save_dir, f"{new_file_name}.txt")
            cv2.imwrite(image_to_save_file_name, image_to_save)

            with open(label_to_save_file_name, "w") as txt:
                for label in dic_labels[new_file_name].values():
                    txt.write(f"{label}\n")
                # print(f"{new_file_name} saved!")
        num += 1
        return dic_labels, file_name_map


if __name__ == "__main__":
    data_folder = DATA_DIR.joinpath('radar1')
    labelme_to_yolo_legacy(data_folder)