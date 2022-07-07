import decimal
import glob
import os
import random

import cv2
import numpy as np
from scipy.ndimage import affine_transform
from utils.common.project_paths import GetPaths

paths = GetPaths()
read_image_path = paths.get_data_folder("data_for_train", "images", "train")
read_label_path = paths.get_data_folder("data_for_train", "labels", "train")

image_list = sorted(glob.glob(os.path.join(read_image_path, "*.png")))
label_list = sorted(glob.glob(os.path.join(read_label_path, "*.txt")))


def ratio_to_coordinate(img_shape, label_ratio_centre):
    """ 라벨의 비율에서 좌표로 변환하는 함수 입니다.

    :param img_shape: 이미지의 사이즈를 입력하는 함수입니다.
    :param label_ratio_centre:
    :return:
    """
    type_tuple = type(tuple([0]))

    if type(img_shape) == type_tuple:
        img_shape_x, img_shape_y = img_shape
    else:
        img_shape_y, img_shape_x = img_shape.shape

    label_ratio_centre_x, label_ratio_centre_y = label_ratio_centre

    label_coordinate_centre_x = round(label_ratio_centre_x * img_shape_x)
    label_coordinate_centre_y = round(label_ratio_centre_y * img_shape_y)
    return label_coordinate_centre_x, label_coordinate_centre_y


def coordinate_to_ratio(img_shape, label_coordinate_centre):
    """
    라벨 좌표에서 비율로 변환하는 함수
    :param img_shape:
    :param label_coordinate_centre:
    :return:
    """
    type_tuple = type(tuple([0]))
    if type(img_shape) == type_tuple:
        img_shape_x, img_shape_y = img_shape
    else:
        img_shape_y, img_shape_x = img_shape.shape

    label_coordinate_centre_x, label_coordinate_centre_y = label_coordinate_centre

    label_ratio_centre_x = label_coordinate_centre_x / img_shape_x
    label_ratio_centre_y = label_coordinate_centre_y / img_shape_y
    return label_ratio_centre_x, label_ratio_centre_y


def load():
    """

    :return:
    """
    for IMAGE, LABEL in zip(image_list, label_list):
        file_name = os.path.split(IMAGE)[1].split(".")[0]
        print(os.path.split(IMAGE)[1].split(".")[0])

        source = cv2.imread(IMAGE, 0)
        img = source.copy()

        img_shape_y, img_shape_x = img.shape
        img_centre = (round(img_shape_x / 2), round(img_shape_y / 2))

        with open(LABEL, "r") as f:
            lines = f.readlines()

        dic_cod_before = {}
        for idx, line in enumerate(lines):
            str_class, str_centre_x, str_centre_y, str_width, str_height = line.split()
            label_Class = int(str_class)
            label_ratio_centre_x = decimal.Decimal(str_centre_x)
            label_ratio_centre_y = decimal.Decimal(str_centre_y)
            label_ratio_centre = (label_ratio_centre_x, label_ratio_centre_y)

            label_centre_x, label_centre_y = ratio_to_coordinate(
                img, label_ratio_centre
            )

            # euclidean_distance = r = math.sqrt((label_centre_x - img_centre_x)**2 + (label_centre_y - img_centre_y)**2)
            dic_cod_before[idx] = [
                label_Class,
                label_centre_x,
                label_centre_y,
                str_width,
                str_height,
            ]

        for rotate_num in range(5):
            rotate_theta = random.randint(-180, 180)
            # rotate_theta = 45
            print(f"{rotate_theta} 회전")
            rotation_matrix = cv2.getRotationMatrix2D(img_centre, rotate_theta, 1)
            img_rotate = affine_transform(img, rotation_matrix)
            label_sparse_matrix = np.zeros(img_rotate.shape)
            cv2.imwrite(
                os.path.join(read_image_path, f"{file_name}_rotate{rotate_num}.png"),
                img_rotate,
            )

            with open(
                os.path.join(read_label_path, f"{file_name}_rotate{rotate_num}.txt"),
                "w",
            ) as f:
                for key, value in dic_cod_before.items():
                    label_Class = value[0]
                    label_centre_x = value[1]
                    label_centre_y = value[2]
                    WIDTH = value[3]
                    HEIGHT = value[4]

                    rotate_x = round(
                        label_centre_x * rotation_matrix[0][0]
                        + label_centre_y * rotation_matrix[0][1]
                        + rotation_matrix[0][2]
                    )
                    rotate_y = round(
                        label_centre_x * rotation_matrix[1][0]
                        + label_centre_y * rotation_matrix[1][1]
                        + rotation_matrix[1][2]
                    )
                    if 0 < rotate_x <= img_shape_x and 0 < rotate_y <= img_shape_y:

                        (
                            label_ratio_centre_x,
                            label_ratio_centre_y,
                        ) = coordinate_to_ratio(
                            (img_shape_x, img_shape_y), (rotate_x, rotate_y)
                        )
                        dic_cod_after = f"{label_Class} {label_ratio_centre_x} {label_ratio_centre_y} {WIDTH} {HEIGHT}"
                        f.write(f"{dic_cod_after}\n")


if __name__ == "__main__":
    load()
