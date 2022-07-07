from utils.inference.inference_modules import InferenceModules
import os
import cv2
import json
from tqdm import tqdm
from utils.common.project_paths import GetPaths

class TestInference(InferenceModules):
    def __init__(self):
        super().__init__()
        self.paths = GetPaths()
        self.test_dir = self.paths.get_project_root('test_inference')
        self.test_data_dir = self.paths.get_project_root('test_inference', 'test_data')
        self.test_result_dir = self.paths.get_project_root('test_inference', 'test_results')
        self.data_list, self.circle_information, self.model = self.load()

    def load(self):
        data_list = sorted(self.get_data_list(path=self.test_data_dir))
        circle_information = self.load_circle_coordinate_dict('test_inference/test_circle_coordinates_old.json')
        weight = self.load_weight(weight='weights/old_radar_gray/weights/best.pt')
        model = self.load_model(weight=weight)
        return data_list, circle_information, model

    def process(self):
        save_dir_ = self.test_result_dir
        os.makedirs(save_dir_, exist_ok=True)
        print(f'결과가 "{save_dir_}" 에 저장됩니다.')

        for data in tqdm(self.data_list, desc='inference'):
            data_file_name = os.path.split(data)[1]
            original_image, image_to_inference, output = self.inference(
                self.model, data, self.circle_information
            )
            xyxyn = self.ratio_to_coordinate(output)  # 라벨들의 xy 좌표의 비율, class, class_name을 가지는 dict로 반환
            original_xyxyn = self.crop_to_original(xyxyn, self.circle_information)  # 자른 원형의 좌표를 자르기 전 좌표로 변환
            labeled_image = self.display(original_image, original_xyxyn, show=True)

            image_path = os.path.join(save_dir_, data_file_name)
            label_path = f"{image_path.split('.')[0]}.json"
            cv2.imwrite(image_path, labeled_image)
            with open(label_path, "w") as f:
                json.dump(original_xyxyn, f)

            # print(f"{idx + 1}/{len(self.data_list)} completed")

def main():
    test = TestInference()
    test.process()

if __name__ == '__main__':
    main()

