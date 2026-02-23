import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[2]
sys.path.append(str(ROOT_DIR))

import glob
import json
import shutil

from tqdm import tqdm
import random


class PreProcessModules:
    @staticmethod
    def stratified_split(source_dirs, output_dir, ratio=(0.8, 0.1, 0.1), seed=42):
        """radar 종류별 층화추출로 train/valid/test 분할

        :param source_dirs: 기존 YOLO 형식 데이터가 있는 폴더 리스트 (PNG+TXT)
        :param output_dir: 분할 결과를 저장할 폴더
        :param ratio: (train, valid, test) 비율
        :param seed: 랜덤 시드
        """
        random.seed(seed)

        # 1. 모든 소스에서 고유한 PNG 파일 수집
        all_files = {}
        for src_dir in source_dirs:
            for png_file in glob.glob(os.path.join(src_dir, "*.png")):
                fname = os.path.basename(png_file)
                if fname not in all_files:
                    all_files[fname] = png_file

        # 2. radar 종류별로 그룹핑 (파일명 prefix 기준)
        groups = {}
        for fname, fpath in all_files.items():
            prefix = '_'.join(fname.split('_')[:2])  # new_radar / old_radar
            groups.setdefault(prefix, []).append(fname)

        # 3. 출력 폴더 생성
        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        # 4. 그룹별 층화추출 분할
        train_ratio, valid_ratio, test_ratio = ratio
        summary = {}

        for prefix, fnames in sorted(groups.items()):
            random.shuffle(fnames)
            n = len(fnames)
            n_train = int(n * train_ratio)
            n_valid = int(n * valid_ratio)

            splits = {
                'train': fnames[:n_train],
                'valid': fnames[n_train:n_train + n_valid],
                'test': fnames[n_train + n_valid:],
            }
            summary[prefix] = {k: len(v) for k, v in splits.items()}

            for split_name, split_fnames in splits.items():
                dst = os.path.join(output_dir, split_name)
                for fname in tqdm(split_fnames, desc=f'{prefix} → {split_name}'):
                    src_png = all_files[fname]
                    src_txt = src_png.replace('.png', '.txt')
                    shutil.copy2(src_png, os.path.join(dst, fname))
                    txt_fname = fname.replace('.png', '.txt')
                    if os.path.exists(src_txt):
                        shutil.copy2(src_txt, os.path.join(dst, txt_fname))

        # 5. 결과 출력
        print("\n=== 층화추출 분할 결과 ===")
        for prefix, counts in summary.items():
            total = sum(counts.values())
            print(f"  {prefix}: 총 {total}장 → train {counts['train']} / valid {counts['valid']} / test {counts['test']}")
        total_all = sum(sum(c.values()) for c in summary.values())
        print(f"  전체: {total_all}장")
        return summary

    @staticmethod
    def convert_raw_to_yolo(raw_dirs, output_dir, seed=42, ratio=(0.8, 0.1, 0.1)):
        """raw LabelMe 데이터를 YOLO 형식으로 변환 + 층화추출 분할

        :param raw_dirs: {'radar_name': 'path/to/raw_dir'} 형태의 딕셔너리
        :param output_dir: 결과 저장 폴더 (train/valid/test 하위 생성)
        :param seed: 랜덤 시드
        :param ratio: (train, valid, test) 비율
        """
        # 라벨 정리 맵
        except_labels = {'range', 'object', 'terra'}
        label_fix = {
            'boat': 'boat_triangle',
            'boat_triangle': 'boat_triangle',
            'boat triangle': 'boat_triangle',
            'boat_rectangle': 'boat_rectangle',
            'start': 'start',
            'end': 'end',
        }
        class_map = {
            'boat_rectangle': 0,
            'boat_triangle': 1,
            'end': 2,
            'start': 3,
        }

        # 1. 변환: radar별로 임시 폴더에 저장
        tmp_dirs = {}
        for radar_name, raw_dir in raw_dirs.items():
            tmp_dir = os.path.join(output_dir, f'_tmp_{radar_name}')
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_dirs[radar_name] = tmp_dir

            json_files = sorted(glob.glob(os.path.join(raw_dir, '*.json')))
            png_files = sorted(glob.glob(os.path.join(raw_dir, '*.png')))

            # JSON과 PNG 매칭
            png_set = {Path(p).stem for p in png_files}
            paired = [(j, os.path.join(raw_dir, f'{Path(j).stem}.png'))
                      for j in json_files if Path(j).stem in png_set]

            for json_file, png_file in tqdm(paired, desc=f'{radar_name} 변환'):
                stem = Path(json_file).stem

                with open(json_file, 'r') as f:
                    data = json.load(f)

                img_w = data['imageWidth']
                img_h = data['imageHeight']
                shapes = data['shapes']

                lines = []
                for shape in shapes:
                    raw_label = shape['label']
                    if raw_label.replace(' ', '_').lower() in except_labels:
                        continue

                    fixed = label_fix.get(raw_label)
                    if fixed is None:
                        fixed = label_fix.get(raw_label.replace(' ', '_'))
                    if fixed is None:
                        continue

                    cls_id = class_map[fixed]
                    p1, p2 = shape['points']
                    x_center = round((p1[0] + p2[0]) / 2 / img_w, 6)
                    y_center = round((p1[1] + p2[1]) / 2 / img_h, 6)
                    w = round(abs(p2[0] - p1[0]) / img_w, 6)
                    h = round(abs(p2[1] - p1[1]) / img_h, 6)
                    lines.append(f'{cls_id} {x_center} {y_center} {w} {h}')

                # 이미지 복사 + 라벨 저장
                dst_png = os.path.join(tmp_dir, f'{radar_name}_{stem}.png')
                dst_txt = os.path.join(tmp_dir, f'{radar_name}_{stem}.txt')
                shutil.copy2(png_file, dst_png)
                with open(dst_txt, 'w') as f:
                    f.write('\n'.join(lines) + '\n' if lines else '')

        # 2. 층화추출 분할
        random.seed(seed)
        train_ratio, valid_ratio, _ = ratio

        for split in ['train', 'valid', 'test']:
            os.makedirs(os.path.join(output_dir, split), exist_ok=True)

        summary = {}
        for radar_name, tmp_dir in tmp_dirs.items():
            pngs = sorted(glob.glob(os.path.join(tmp_dir, '*.png')))
            fnames = [os.path.basename(p) for p in pngs]
            random.shuffle(fnames)

            n = len(fnames)
            n_train = int(n * train_ratio)
            n_valid = int(n * valid_ratio)

            splits = {
                'train': fnames[:n_train],
                'valid': fnames[n_train:n_train + n_valid],
                'test': fnames[n_train + n_valid:],
            }
            summary[radar_name] = {k: len(v) for k, v in splits.items()}

            for split_name, split_fnames in splits.items():
                dst = os.path.join(output_dir, split_name)
                for fname in tqdm(split_fnames, desc=f'{radar_name} → {split_name}'):
                    src_png = os.path.join(tmp_dir, fname)
                    src_txt = src_png.replace('.png', '.txt')
                    shutil.move(src_png, os.path.join(dst, fname))
                    if os.path.exists(src_txt):
                        shutil.move(src_txt, os.path.join(dst, fname.replace('.png', '.txt')))

            # 임시 폴더 삭제
            shutil.rmtree(tmp_dir)

        # 3. 결과 출력
        print('\n=== 데이터셋 생성 결과 ===')
        print(f'클래스: {class_map}')
        for radar_name, counts in summary.items():
            total = sum(counts.values())
            print(f'  {radar_name}: 총 {total}장 → train {counts["train"]} / valid {counts["valid"]} / test {counts["test"]}')
        total_all = sum(sum(c.values()) for c in summary.values())
        print(f'  전체: {total_all}장')
        return class_map, summary


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / 'data'

    raw_dirs = {
        'radar1': str(data_dir / 'radar1'),
        'radar2': str(data_dir / 'radar2'),
    }
    output_dir = str(data_dir / 'train_data' / 'raw_stratified')

    PreProcessModules.convert_raw_to_yolo(
        raw_dirs=raw_dirs,
        output_dir=output_dir,
        ratio=(0.8, 0.1, 0.1),
        seed=42,
    )
