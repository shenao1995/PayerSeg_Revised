import sys
import os
import time
import argparse
from multiprocessing import Process

# 设置环境变量，减少TF日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(sys.path[0]))

import other.preprocess_overlap_cropped as preprocess_overlap_cropped
from main_spine_localization_overlap_cropped import get_test_bounding_box
from main_vertebrae_localization_overlap_cropped import vertebrae_localization
from main_vertebrae_segmentation_overlap_cropped import vertebrae_segmentation


def parse_args():
    parser = argparse.ArgumentParser(description='Spine Segmentation Pipeline')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to raw input images')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save results')
    parser.add_argument('--model_step1', type=str, required=True, help='Path to Step 1 model weights')
    parser.add_argument('--model_step2', type=str, required=True, help='Path to Step 2 model weights')
    parser.add_argument('--model_step3', type=str, required=True, help='Path to Step 3 model weights')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # 确保输出目录存在
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    total_time_start = time.time()

    print('--- Starting Preprocessing ---')
    # 预处理：将原始图像重定向并保存到 output_folder
    # 注意：后续步骤将从 output_folder 读取处理后的图像
    preprocess_overlap_cropped.roi_before(args.input_folder, args.output_folder)
    print('Preprocessing successful')

    print('--- Starting Step 1: Spine Localization ---')
    # 使用 Process 确保显存释放
    p1 = Process(target=get_test_bounding_box, args=(args.output_folder, args.output_folder, args.model_step1))
    p1.start()
    p1.join()
    p1.close()
    print('Step 1 completed')

    print('--- Starting Step 2: Vertebrae Localization ---')
    p2 = Process(target=vertebrae_localization, args=(args.output_folder, args.output_folder, args.model_step2))
    p2.start()
    p2.join()
    p2.close()
    print('Step 2 completed')

    print('--- Starting Step 3: Vertebrae Segmentation ---')
    p3 = Process(target=vertebrae_segmentation, args=(args.output_folder, args.output_folder, args.model_step3))
    p3.start()
    p3.join()
    p3.close()
    print('Step 3 completed')
    print("Total pipeline time:", time.time() - total_time_start)
