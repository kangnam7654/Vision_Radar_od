import os
from pathlib import Path


class GetPaths:
    """ 프로젝트 내부의 각 경로에 접근할 수 있는 클래스입니다.

    """
    def __init__(self):
        self.project_root = self.get_project_root()
        self.data_folder = self.get_data_folder()
        self.circle_coordinate_folder = self.get_circle_coordinate_folder()
        self.yaml_folder = self.get_yaml_folder()

    @staticmethod
    def get_project_root(*paths):
        """ 프로젝트의 최상위 Root 의 경로를 받아오는 함수입니다.

        :param paths: 프로젝트 최상위 Root 부터의 추가적인 경로 입력하는 변수입니다.
        :return: 프로젝트의 root 경로를 반환
        """
        project_root_path = os.path.join(Path(__file__).parents[2], *paths)
        return project_root_path

    def get_data_folder(self, *paths):
        """ 프로젝트내의 data 폴더의 경로를 받아오는 함수입니다.

        :param paths: data 폴더부터 시작하여 추가적인 경로를 입력하는 변수입니다.
        :return: data 폴더의 경로를 반환
        """
        data_folder_path = self.get_project_root("data", *paths)
        return data_folder_path

    def get_circle_coordinate_folder(self, *paths):
        """ 프로젝트내의 coordinate 폴더의 경로를 받아오는 함수입니다.

        :param paths: coordinate 부터 시작하여 추가적인 경로를 입력하는 변수입니다.
        :return: coordinate 폴더의 경로를 반환
        """
        circle_coordinate_path = self.get_project_root("circle_coordinate", *paths)
        return circle_coordinate_path

    def get_yaml_folder(self, *paths):
        yaml_path = self.get_project_root('yaml', *paths)
        return yaml_path

    @staticmethod
    def curdir():
        """ os.path.curdir 와 같은 함수로 함수 호출 위치의 directory 를 반환합니다.

        :return:
        """
        return os.path.curdir


if __name__ == "__main__":
    a = GetPaths().get_project_root()
