#!/usr/bin/python
from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf
import utils.io.image
import utils.io.text
from dataset_overlap_cropped import Dataset
from network import Unet
from spine_localization_postprocessing import bb
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm
import multiprocessing
from tensorflow.keras import mixed_precision


class MainLoop(MainLoopBase):
    def __init__(self, cv, config, input_folder, output_folder):
        super().__init__()
        self.cv = cv
        self.config = config
        self.batch_size = 1
        self.learning_rate = config.learning_rate
        self.max_iter = 10000
        self.data_format = 'channels_first'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              dropout_ratio=config.dropout_ratio,
                                              num_levels=config.num_levels,
                                              heatmap_initialization=True,
                                              data_format=self.data_format)
        if config.model == 'unet':
            self.network = Unet

        self.input_folder = input_folder
        self.base_output_folder = output_folder
        self.image_size = [None, None, None]
        self.image_spacing = [config.spacing] * 3
        self.save_output_images = True

    def init_model(self):
        self.model = self.network(num_labels=1, **self.network_parameters)

    def init_optimizer(self):
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter,
                                                                                     0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder,
                                                         model_name=self.model.name,
                                                         cv=str(self.cv),
                                                         additional_info=self.config.info)

    def init_datasets(self):
        dataset_parameters = dict(
            image_base_folder=self.input_folder,
            image_size=self.image_size,
            image_spacing=self.image_spacing,
            normalize_zero_mean_unit_variance=False,
            valid_output_sizes_x=[32, 64, 96, 128],
            valid_output_sizes_y=[32, 64, 96, 128],
            valid_output_sizes_z=[32, 64, 96, 128],
            use_variable_image_size=True,
            cv=self.cv,
            input_gaussian_sigma=0.75,
            output_image_type=np.float32,
            data_format=self.data_format,
            save_debug_images=False,
        )
        dataset = Dataset(**dataset_parameters)
        self.dataset_val = dataset.dataset_val()

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        image = np.expand_dims(generators['image'], axis=0)
        prediction = self.model(image, training=False)
        prediction = np.squeeze(prediction, axis=0)
        transformation = transformations['image']
        image = generators['image']
        return image, prediction, transformation

    def test(self):
        print('Testing Spine Localization...')
        num_entries = self.dataset_val.num_entries()
        bbs = {}
        for _ in tqdm(range(num_entries), desc='Localization'):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            image, prediction, transformation = self.test_full_image(dataset_entry)
            start_transformed, end_transformed = bb(prediction, transformation, self.image_spacing)
            bbs[current_id] = start_transformed + end_transformed

            if self.save_output_images:
                # 【修改】为每个样本创建独立文件夹: Results/Filename/step1_spine_localization
                sample_dir = os.path.join(self.base_output_folder, current_id, 'step1_spine_localization')
                if not os.path.exists(sample_dir):
                    os.makedirs(sample_dir)

                origin = transformation.TransformPoint(np.zeros(3, np.float64))

                # 保存到新路径
                utils.io.image.write_multichannel_np(image,
                                                     os.path.join(sample_dir, current_id + '_input.mha'),
                                                     output_normalization_mode='min_max',
                                                     sitk_image_output_mode='vector',
                                                     data_format=self.data_format,
                                                     image_type=np.uint8,
                                                     spacing=self.image_spacing,
                                                     origin=origin)
                utils.io.image.write_multichannel_np(prediction,
                                                     os.path.join(sample_dir, current_id + '_prediction.mha'),
                                                     output_normalization_mode=(0, 1),
                                                     sitk_image_output_mode='vector',
                                                     data_format=self.data_format,
                                                     image_type=np.uint8,
                                                     spacing=self.image_spacing,
                                                     origin=origin)

        # 【注意】bbs.csv 仍保存在根目录，因为后续步骤需要统一读取它
        utils.io.text.save_dict_csv(bbs, os.path.join(self.base_output_folder, 'bbs.csv'))


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def run_inference_step1(config_dict, input_folder, output_folder, model_path):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    config = dotdict(config_dict)

    with MainLoop(0, config, input_folder, output_folder) as loop:
        loop.load_model_filename = model_path
        loop.run_test()


def get_test_bounding_box(input_folder, output_folder, model_path):
    config = {
        'num_filters_base': 96,
        'dropout_ratio': 0.25,
        'activation': 'lrelu',
        'learning_rate': 0.0001,
        'model': 'unet',
        'num_levels': 5,
        'spacing': 8.0,
        'cv': 0,
        'info': 'd0_25_fin'
    }

    p = multiprocessing.Process(target=run_inference_step1,
                                args=(config, input_folder, output_folder, model_path))
    p.start()
    p.join()
    p.close()
