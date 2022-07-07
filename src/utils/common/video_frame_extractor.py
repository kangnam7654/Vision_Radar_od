import glob
import os

import cv2
from utils.common.project_paths import GetPaths


class VideoFrameExtract:
    """ 영상에서 학습용 데이터를 추출하기 위한 클래스입니다.
    영상에서 일정 시간(Frame 단위) 마다 프레임을 저장합니다.
    """

    def __init__(self, file, interval_sec, dir_name=''):
        """
        :param file: 영상의 경로를 입력합니다.
        :param interval_sec: 원본 재생 속도 기준으로 몇초에 한장씩 저장할 것인지 설정하는 함수입니다. 단위는 s 입니다.
        :param dir_name: 저장할 directory 의 folder name 입니다.
        """
        self.cap = cv2.VideoCapture(file)
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.interval = round(self.original_fps * interval_sec)
        self.save_dir = os.path.join(GetPaths().project_root, "data", dir_name)
        os.makedirs(self.save_dir, exist_ok=True)

    def play(self, show=False):
        """ 영상을 재생하여 추출하는 메서드입니다.

        작업시간 단축을 위하여 영상속도는 최대로 고정됩니다.
        :param show: 영상의 show 를 할지 말지 결정하는 변수입니다.
        :return:
        """
        file_num = 0
        cnt = 0
        while True:
            cnt += 1
            retval, frame = self.cap.read()
            if not retval:
                raise Exception(f"return value: {retval}")

            if cnt % self.interval == 0:
                file_name = f"{str(file_num).zfill(6)}.png"  # 파일 Naming 000000.png -> 000001.png 순으로 저장
                file_path = os.path.join(self.save_dir, file_name)
                cv2.imwrite(file_path, frame)
                print(f"{file_name} saved")
                file_num += 1

            if show:
                cv2.imshow("video", frame)
                cv2.waitKey(1)

    def make_video(self, show=False):
        cnt = 0
        saved_frame = 0
        w = round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')


        fn = 0
        fn_ = str(fn).zfill(4)
        file_name = f'/Users/kimkangnam/Desktop/Project/CompanyProject/DataVoucher/Mapsea/data/video/20211001_K_{fn_}.mp4'
        out = cv2.VideoWriter(file_name, fourcc, self.original_fps, (w, h))

        while True:
            cnt += 1
            retval, frame = self.cap.read()
            if not retval:
                raise Exception(f"return value: {retval}")

            if cnt % self.interval == 0:
                out.write(frame)
                saved_frame += 1
                print(f'frame_ex, saved: {saved_frame}')

            if saved_frame >= 30:
                out.release()
                print(f"{file_name} saved")
                fn_ = str(fn).zfill(4)
                file_name = f'/Users/kimkangnam/Desktop/Project/CompanyProject/DataVoucher/Mapsea/data/video/20211001_K_{fn_}.mp4'
                out = cv2.VideoWriter(file_name, fourcc, self.original_fps, (w, h))

                saved_frame = 0
                fn += 1

            if show:
                cv2.imshow("video", frame)
                cv2.waitKey(1)


if __name__ == "__main__":
    root_dir = GetPaths().project_root
    file_list = sorted(glob.glob(os.path.join(root_dir, "data", "2021.11.02", "*.mp4")))
    num = 0
    extractor = VideoFrameExtract(file=file_list[num], interval_sec=1)
    extractor.make_video(show=True)