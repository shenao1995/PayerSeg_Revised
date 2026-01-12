#!/usr/bin/python
import os
import sys
import traceback
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np
import tensorflow as tf
import utils.io.image
import utils.io.text
import utils.sitk_image
import utils.sitk_np
import utils.np_image
from dataset_overlap_cropped import Dataset
from network import Unet
from tensorflow.keras import mixed_precision
from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm
import multiprocessing
from utils.clear_image_transparency_edge_and_merge import merge_top_and_bottom_image
from utils.get_full_image import get_full_image
from utils.save_read_dict import read_dict, load_test_nii_txt


class MainLoop(MainLoopBase):
    def __init__(self, cv, config, image_folder, setup_folder, output_folder, model_path):
        super().__init__()
        self.use_mixed_precision = False
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)

        self.image_folder = image_folder
        self.setup_base_folder = setup_folder
        self.final_output_folder = output_folder
        self.model_path = model_path

        id_list_file_name = os.path.join(self.image_folder, "test_nii_name.txt")
        if not os.path.exists(id_list_file_name):
            raise FileNotFoundError(f"test_nii_name.txt not found at: {id_list_file_name}")
        self.test_id_list = load_test_nii_txt(id_list_file_name)
        print(f"[DEBUG] Loaded {len(self.test_id_list)} IDs from test_nii_name.txt")

        self.valid_landmarks_file = os.path.join(self.setup_base_folder, 'valid_landmarks.csv')
        if os.path.exists(self.valid_landmarks_file):
            self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file, squeeze=False)
            print(f"[DEBUG] Loaded valid_landmarks.csv with {len(self.valid_landmarks)} cases.")
        else:
            print(f"[WARNING] valid_landmarks.csv not found at {self.valid_landmarks_file}. Step 2 might have failed.")
            self.valid_landmarks = {}

        self.cropped_images_dict = read_dict(self.image_folder)

        self.cv = cv
        self.config = config
        self.batch_size = 1
        self.learning_rate = config.learning_rate
        self.max_iter = 50000
        self.num_labels = 1
        self.num_labels_all = 27
        self.data_format = 'channels_first'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              dropout_ratio=config.dropout_ratio,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        if config.model == 'unet':
            self.network = Unet

        self.image_size = [128, 128, 96]
        self.image_spacing = [config.spacing] * 3

        self.additional_output_folder_info = config.info

        # 【修复】：删除了会导致 AttributeError 的 call_model_and_loss 初始化代码
        # 在推理模式下不需要预编译损失函数。

        self.landmark_labels = [i + 1 for i in range(25)] + [28]
        self.landmark_mapping = dict([(i, self.landmark_labels[i]) for i in range(26)])

    def init_model(self):
        self.norm_moving_average = tf.Variable(1.0)
        self.model = self.network(num_labels=self.num_labels, **self.network_parameters)

    def init_optimizer(self):
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate, amsgrad=True)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler('./output/vertebrae_segmentation_temp',
                                                         model_name=self.model.name,
                                                         cv=str(self.cv),
                                                         additional_info=self.additional_output_folder_info)

    def init_datasets(self):
        landmarks_file = os.path.join(self.setup_base_folder, 'landmarks.csv')
        dataset_parameters = dict(
            image_base_folder=self.image_folder,
            setup_base_folder=self.setup_base_folder,
            base_folder=self.setup_base_folder,
            landmarks_file=landmarks_file,
            image_size=self.image_size,
            image_spacing=self.image_spacing,
            normalize_zero_mean_unit_variance=False,
            cv=self.cv,
            input_gaussian_sigma=0.75,
            heatmap_sigma=3.0,
            generate_single_vertebrae_heatmap=True,
            output_image_type=np.float32,
            data_format=self.data_format,
            save_debug_images=False,
            vertebrae_segmentation_eval=False)

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']

        if 'single_heatmap' in generators:
            heatmap_key = 'single_heatmap'
        elif 'heatmap' in generators:
            heatmap_key = 'heatmap'
        else:
            raise KeyError("Neither 'single_heatmap' nor 'heatmap' found in generators.")

        image = np.expand_dims(generators['image'], axis=0)
        single_heatmap = np.expand_dims(generators[heatmap_key], axis=0)

        image_heatmap_concat = tf.concat([image, single_heatmap],
                                         axis=1 if self.data_format == 'channels_first' else -1)

        prediction = self.model(image_heatmap_concat, training=False)
        prediction = np.squeeze(prediction, axis=0)
        transformation = transformations['image']
        image = generators['image']
        return image, prediction, transformation

    def test(self):
        tqdm.write('Starting Segmentation and Merging...')
        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        filter_largest_cc = True
        cropped_image_path_dict = {}

        for image_id in tqdm(self.test_id_list, desc='Segmenting'):
            try:
                # 1. 检查有效 Landmark
                if image_id not in self.valid_landmarks:
                    tqdm.write(f"[SKIP] {image_id}: Not in valid_landmarks.csv.")
                    continue

                landmarks = self.valid_landmarks[image_id]
                # load_dict_csv 返回可能是列表或字符串
                if not isinstance(landmarks, list):
                    landmarks = [landmarks]
                # 过滤空字符串
                landmarks = [l for l in landmarks if str(l).strip() != '']

                if not landmarks:
                    tqdm.write(f"[SKIP] {image_id}: No valid landmarks found.")
                    continue

                # print(f"Processing {image_id} with {len(landmarks)} vertebrae...")

                first = True
                combined_labels_np = None
                full_image = None

                # 创建独立的输出文件夹
                case_output_dir = os.path.join(self.final_output_folder, image_id, 'step3_vertebrae_segmentation')
                if not os.path.exists(case_output_dir):
                    os.makedirs(case_output_dir)

                # 2. 逐个椎体分割
                for landmark_id in landmarks:
                    try:
                        dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id': landmark_id})
                    except Exception as e:
                        tqdm.write(f"[ERROR] Loading data for {image_id} L{landmark_id}: {e}")
                        continue

                    if first:
                        input_image_sitk = dataset_entry['datasources']['image']
                        full_image = get_full_image(image_id, self.cropped_images_dict, input_image_sitk)
                        combined_labels_np = np.zeros(list(reversed(full_image.GetSize())), dtype=np.uint8)
                        first = False

                    image, prediction, transformation = self.test_full_image(dataset_entry)
                    del dataset_entry

                    # 3. 连通域过滤 (处理维度问题)
                    if filter_largest_cc:
                        pred_np = prediction
                        # 压缩维度直到 3D
                        while pred_np.ndim > 3:
                            pred_np = np.squeeze(pred_np, axis=0)

                        pred_thresh = (pred_np > 0.5).astype(np.uint8)
                        # 使用修复后的 largest_connected_component
                        mask_3d = utils.np_image.largest_connected_component(pred_thresh)

                        # 恢复维度
                        mask_expanded = mask_3d
                        while mask_expanded.ndim < prediction.ndim:
                            mask_expanded = np.expand_dims(mask_expanded, axis=0)

                        prediction = prediction * mask_expanded

                    prediction = prediction.astype(np.float32)

                    # 重采样回原图空间
                    prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(
                        output_image=prediction,
                        output_spacing=self.image_spacing,
                        channel_axis=channel_axis,
                        input_image_sitk=full_image,
                        transform=transformation,
                        interpolator='linear',
                        output_pixel_type=sitk.sitkFloat32
                    )
                    del prediction

                    prediction_resampled_np = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])
                    prediction_thresh_np = (prediction_resampled_np > 0.5).astype(np.uint8)

                    current_label_val = self.landmark_mapping[int(landmark_id)]
                    combined_labels_np[prediction_thresh_np == 1] = current_label_val

                    del prediction_resampled_sitk, prediction_resampled_np

                # 4. 保存结果
                if combined_labels_np is not None:
                    combined_sitk = utils.sitk_np.np_to_sitk(combined_labels_np)
                    combined_sitk.CopyInformation(full_image)

                    seg_filename = os.path.join(case_output_dir, image_id + '_seg.nii.gz')
                    utils.io.image.write(combined_sitk, seg_filename)
                    # tqdm.write(f"[SUCCESS] Saved segmentation: {seg_filename}")

                    if image_id.endswith("_top") or image_id.endswith("_bottom"):
                        if image_id not in cropped_image_path_dict:
                            cropped_image_path_dict[image_id] = seg_filename
                else:
                    tqdm.write(f"[WARNING] No labels generated for {image_id}.")

            except Exception as e:
                tqdm.write(f"[CRITICAL ERROR] Failed processing case {image_id}:")
                traceback.print_exc()
                continue

        # 5. 合并 Cropped 图像
        # tqdm.write("Checking for cropped images to merge...")
        for cropped_image_name in list(self.cropped_images_dict.keys()):
            # 兼容布尔值和字符串 "True"
            is_cropped = str(self.cropped_images_dict[cropped_image_name]).strip().lower() == "true"

            if is_cropped:
                top_id = cropped_image_name + "_top"
                bottom_id = cropped_image_name + "_bottom"

                if top_id in cropped_image_path_dict and bottom_id in cropped_image_path_dict:
                    tqdm.write(f"Merging Top/Bottom for {cropped_image_name}...")
                    merged_case_dir = os.path.join(self.final_output_folder, cropped_image_name,
                                                   'step3_vertebrae_segmentation')
                    if not os.path.exists(merged_case_dir):
                        os.makedirs(merged_case_dir)

                    merged_seg_path = os.path.join(merged_case_dir, cropped_image_name + '_seg.nii.gz')
                    try:
                        merge_top_and_bottom_image(cropped_image_path_dict[top_id], cropped_image_path_dict[bottom_id],
                                                   output_path=merged_seg_path)
                        tqdm.write(f"[SUCCESS] Saved FULL merged segmentation: {merged_seg_path}")
                    except Exception as e:
                        tqdm.write(f"[ERROR] Failed merging {cropped_image_name}: {e}")


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def run_inference_step3(config_dict, input_folder, output_folder, model_path):
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 0:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    config = dotdict(config_dict)
    with MainLoop(0, config, input_folder, output_folder, output_folder, model_path) as loop:
        loop.load_model_filename = model_path
        loop.run_test()


def vertebrae_segmentation(input_folder, output_folder, model_path):
    config = {
        'num_filters_base': 96,
        'dropout_ratio': 0.25,
        'activation': 'lrelu',
        'learning_rate': 0.0001,
        'model': 'unet',
        'num_levels': 5,
        'spacing': 1.0,
        'cv': 0,
        'info': 'd0_25_fin',
        'vertebrae_segmentation_eval': False
    }

    p = multiprocessing.Process(target=run_inference_step3,
                                args=(config, input_folder, output_folder, model_path))
    p.start()
    p.join()
    p.close()