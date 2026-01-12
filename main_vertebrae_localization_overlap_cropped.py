#!/usr/bin/python
import pickle
from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf
import utils.io.text
import utils.io.landmark
import utils.io.image
from dataset_overlap_cropped import Dataset
from network import SpatialConfigurationNet, Unet
from tensorflow.keras import mixed_precision
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm
from utils.image_tiler import ImageTiler, LandmarkTiler
from utils.landmark.common import Landmark
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.spine_postprocessing_graph import SpinePostprocessingGraph
from utils.landmark.visualization.landmark_visualization_matplotlib import LandmarkVisualizationMatplotlib
from vertebrae_localization_postprocessing import add_landmarks_from_neighbors, filter_landmarks_top_bottom, \
    reshift_landmarks
import multiprocessing
from utils.save_read_dict import read_dict
from utils.get_full_image import get_full_image
from utils.landmark.show_full_image import get_all_landmarks


class MainLoop(MainLoopBase):
    def __init__(self, cv, config, input_folder, output_folder, setup_folder, model_path):
        super().__init__()
        self.cv = cv
        self.config = config

        # 【修复1】开启混合精度，减少显存占用防止 OOM
        self.use_mixed_precision = True
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)

        self.batch_size = 1
        self.learning_rate = config.learning_rate
        self.max_iter = 50000
        self.num_landmarks = 26
        self.heatmap_sigma = config.heatmap_sigma
        self.learnable_sigma = config.learnable_sigma
        self.data_format = 'channels_first'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              spatial_downsample=config.spatial_downsample,
                                              dropout_ratio=config.dropout_ratio,
                                              local_activation=config.local_activation,
                                              spatial_activation=config.spatial_activation,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        if config.model == 'scn':
            self.network = SpatialConfigurationNet

        self.input_folder = input_folder
        self.base_output_folder = output_folder
        self.setup_folder = setup_folder
        self.image_size = [None, None, None]
        self.image_spacing = [config.spacing] * 3

        # 【修复2】减小裁剪尺寸，防止 OOM
        self.max_image_size_for_cropped_test = [96, 96, 128]  # 原 [128, 128, 448]
        self.cropped_inc = [0, 64, 0, 0]  # 步长
        self.save_output_images = True
        self.sigma_regularization = 100.0

        # 开启后处理
        self.evaluate_landmarks_postprocessing = True

        self.cropped_images_dict = read_dict(self.input_folder)

    def init_model(self):
        self.sigmas_variables = tf.Variable([self.heatmap_sigma] * self.num_landmarks, name='sigmas', trainable=True)
        self.sigmas = self.sigmas_variables
        if not self.learnable_sigma:
            self.sigmas = tf.stop_gradient(self.sigmas)
        self.model = self.network(num_labels=self.num_landmarks, **self.network_parameters)

    def init_optimizer(self):
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter,
                                                                                     0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule, amsgrad=False)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model,
                                              optimizer=self.optimizer,
                                              sigmas=self.sigmas_variables)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder,
                                                         model_name=self.model.name,
                                                         cv=str(self.cv),
                                                         additional_info=self.config.info)

    def init_datasets(self):
        dataset_parameters = dict(
            image_base_folder=self.input_folder,
            setup_base_folder=self.setup_folder,
            image_size=self.image_size,
            image_spacing=self.image_spacing,
            num_landmarks=self.num_landmarks,
            normalize_zero_mean_unit_variance=False,
            cv=self.cv,
            input_gaussian_sigma=0.75,
            crop_image_top_bottom=True,
            use_variable_image_size=True,
            load_spine_bbs=True,
            valid_output_sizes_x=[64, 96],
            valid_output_sizes_y=[64, 96],
            valid_output_sizes_z=[64, 96, 128, 160, 192, 224, 256, 288, 320],
            translate_to_center_landmarks=True,
            translate_by_random_factor=True,
            data_format=self.data_format,
            save_debug_images=False,
            vertebrae_localization_eval=False)

        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

    def save_valid_landmarks_list(self, landmarks_dict, filename):
        valid_landmarks = {}
        for image_id, landmarks in landmarks_dict.items():
            current_valid_landmarks = []
            for landmark_id, landmark in enumerate(landmarks):
                if landmark.is_valid:
                    current_valid_landmarks.append(landmark_id)
            valid_landmarks[image_id] = current_valid_landmarks
        utils.io.text.save_dict_csv(valid_landmarks, filename)

    def test_cropped_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        transformation = transformations['image']
        full_image = generators['image']

        image_size_for_tilers = np.minimum(full_image.shape[1:],
                                           list(reversed(self.max_image_size_for_cropped_test))).tolist()
        image_size_np = [1] + image_size_for_tilers
        labels_size_np = [self.num_landmarks] + image_size_for_tilers

        image_tiler = ImageTiler(full_image.shape, image_size_np, self.cropped_inc, True, -1)
        prediction_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np, self.cropped_inc,
                                      True, -np.inf)
        prediction_local_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np,
                                            self.cropped_inc, True, -np.inf)
        prediction_spatial_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], labels_size_np,
                                              self.cropped_inc, True, -np.inf)

        for image_tiler, prediction_tiler, prediction_local_tiler, prediction_spatial_tiler in zip(
                image_tiler, prediction_tiler, prediction_local_tiler, prediction_spatial_tiler):
            current_image = image_tiler.get_current_data(full_image)
            prediction, prediction_local, prediction_spatial = self.model(np.expand_dims(current_image, axis=0),
                                                                          training=False)

            image_tiler.set_current_data(current_image)
            prediction_tiler.set_current_data(np.squeeze(prediction, axis=0))
            prediction_local_tiler.set_current_data(np.squeeze(prediction_local, axis=0))
            prediction_spatial_tiler.set_current_data(np.squeeze(prediction_spatial, axis=0))

        return image_tiler.output_image, prediction_tiler.output_image, prediction_local_tiler.output_image, prediction_spatial_tiler.output_image, transformation

    def test(self):
        print('Testing Vertebrae Localization...')
        vis = LandmarkVisualizationMatplotlib(dim=3,
                                              annotations=dict([(i, f'C{i + 1}') for i in range(7)] +
                                                               [(i, f'T{i - 6}') for i in range(7, 19)] +
                                                               [(i, f'L{i - 18}') for i in range(19, 25)] +
                                                               [(25, 'T13')]))

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        heatmap_maxima = HeatmapTest(channel_axis, False, return_multiple_maxima=True, min_max_value=0.05,
                                     smoothing_sigma=2.0)

        # 尝试加载后处理参数，如果失败则跳过后处理
        try:
            with open('possible_successors.pickle', 'rb') as f:
                possible_successors = pickle.load(f)
            with open('units_distances.pickle', 'rb') as f:
                offsets_mean, distances_mean, distances_std = pickle.load(f)
        except Exception as e:
            print(f"Warning: Pickle files not found: {e}. Postprocessing disabled.")
            self.evaluate_landmarks_postprocessing = False

        if self.evaluate_landmarks_postprocessing:
            spine_postprocessing = SpinePostprocessingGraph(num_landmarks=self.num_landmarks,
                                                            possible_successors=possible_successors,
                                                            offsets_mean=offsets_mean,
                                                            distances_mean=distances_mean,
                                                            distances_std=distances_std,
                                                            bias=2.0,
                                                            l=0.2)

        landmarks = {}
        num_entries = self.dataset_val.num_entries()
        full_image_landmarks_dict = {}

        # 根目录文件路径（用于Step3读取）
        root_landmarks_path = os.path.join(self.base_output_folder, 'landmarks.csv')
        root_valid_landmarks_path = os.path.join(self.base_output_folder, 'valid_landmarks.csv')

        for _ in tqdm(range(num_entries), desc='Localization'):
            try:
                dataset_entry = self.dataset_val.get_next()
                current_id = dataset_entry['id']['image_id']
                datasources = dataset_entry['datasources']
                input_image = datasources['image']

                full_image = get_full_image(current_id, self.cropped_images_dict, input_image)

                image, prediction, prediction_local, prediction_spatial, transformation = self.test_cropped_image(
                    dataset_entry)

                local_maxima_landmarks = heatmap_maxima.get_landmarks(prediction, input_image, self.image_spacing,
                                                                      transformation)
                curr_landmarks_no_postprocessing = [
                    l[0] if len(l) > 0 else Landmark(coords=[np.nan] * 3, is_valid=False) for l in
                    local_maxima_landmarks]

                curr_landmarks = curr_landmarks_no_postprocessing

                if self.evaluate_landmarks_postprocessing:
                    try:
                        local_maxima_landmarks = add_landmarks_from_neighbors(local_maxima_landmarks)
                        curr_landmarks = spine_postprocessing.solve_local_heatmap_maxima(local_maxima_landmarks)
                        curr_landmarks = reshift_landmarks(curr_landmarks)
                        curr_landmarks = filter_landmarks_top_bottom(curr_landmarks, full_image)
                    except Exception as e:
                        # 【修复3】解决中文编码错误：用 repr(e) 安全打印错误信息
                        tqdm.write(f"Postprocessing error for {current_id}: {repr(e)}")
                        curr_landmarks = curr_landmarks_no_postprocessing

                landmarks[current_id] = curr_landmarks

                # 【修复4】增量保存：每次处理完一个病例都更新根目录 CSV
                utils.io.landmark.save_points_csv(landmarks, root_landmarks_path)
                self.save_valid_landmarks_list(landmarks, root_valid_landmarks_path)

                # 【修复5】创建子文件夹并保存 per-case 结果
                sample_dir = os.path.join(self.base_output_folder, current_id, 'step2_vertebrae_localization')
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                if self.save_output_images:
                    vis.visualize_landmark_projections(full_image, curr_landmarks,
                                                       filename=os.path.join(sample_dir,
                                                                             current_id + '_landmarks_final.png'))
                    # 保存热图
                    origin = transformation.TransformPoint(np.zeros(3, np.float64))
                    utils.io.image.write_multichannel_np(prediction,
                                                         os.path.join(sample_dir, current_id + '_prediction.mha'),
                                                         output_normalization_mode=(-1, 1),
                                                         sitk_image_output_mode='vector', data_format=self.data_format,
                                                         image_type=np.uint8, spacing=self.image_spacing, origin=origin)

                full_image_landmarks_dict = get_all_landmarks(current_id=current_id,
                                                              curr_landmarks_no_postprocessing=curr_landmarks_no_postprocessing,
                                                              curr_landmarks=curr_landmarks,
                                                              full_image_landmarks_dict=full_image_landmarks_dict,
                                                              cropped_images_dict=self.cropped_images_dict)
            except Exception as e:
                # 捕获循环内的严重错误（如 OOM），避免中断整个流程，并安全打印
                tqdm.write(f"CRITICAL ERROR processing case: {repr(e)}")
                # 显式清理
                tf.keras.backend.clear_session()


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def run_inference_step2(config_dict, input_folder, output_folder, model_path):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    config = dotdict(config_dict)
    # setup_folder 设为 output_folder 以读取 Step 1 的结果
    with MainLoop(0, config, input_folder, output_folder, output_folder, model_path) as loop:
        loop.load_model_filename = model_path
        loop.run_test()


def vertebrae_localization(input_folder, output_folder, model_path):
    config = {
        'num_filters_base': 96,
        'dropout_ratio': 0.25,
        'activation': 'lrelu',
        'spatial_downsample': 4,
        'heatmap_sigma': 3.0,
        'learnable_sigma': False,
        'learning_rate': 0.0001,
        'model': 'scn',
        'local_activation': 'tanh',
        'spatial_activation': 'tanh',
        'num_levels': 4,
        'spacing': 2.0,
        'cv': 0,
        'info': 'fin_cv',
        'vertebrae_localization_eval': False
    }

    p = multiprocessing.Process(target=run_inference_step2,
                                args=(config, input_folder, output_folder, model_path))
    p.start()
    p.join()
    p.close()