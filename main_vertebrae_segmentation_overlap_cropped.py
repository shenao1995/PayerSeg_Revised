#!/usr/bin/python
import os
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
from tensorflow_train_v2.train_loop import MainLoopBase
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tqdm import tqdm
import multiprocessing
from utils.get_full_image import get_full_image
from utils.save_read_dict import read_dict, load_test_nii_txt


class MainLoop(MainLoopBase):
    def __init__(self, cv, config, input_folder, output_folder):
        super().__init__()
        self.cv = cv
        self.config = config
        self.batch_size = 1
        self.learning_rate = config.learning_rate
        self.max_iter = 50000
        self.data_format = 'channels_first'
        self.network_parameters = OrderedDict(num_filters_base=config.num_filters_base,
                                              activation=config.activation,
                                              dropout_ratio=config.dropout_ratio,
                                              num_levels=config.num_levels,
                                              data_format=self.data_format)
        if config.model == 'unet':
            self.network = Unet

        self.input_folder = input_folder
        self.base_output_folder = output_folder
        self.image_size = [128, 128, 96]
        self.image_spacing = [config.spacing] * 3
        self.save_output_images = True

        id_list_file_name = os.path.join(self.input_folder, "test_nii_name.txt")
        self.test_id_list = load_test_nii_txt(id_list_file_name)

        self.valid_landmarks_file = os.path.join(self.base_output_folder, 'valid_landmarks.csv')
        self.valid_landmarks = utils.io.text.load_dict_csv(self.valid_landmarks_file, squeeze=False)

        self.landmark_labels = [i + 1 for i in range(25)] + [28]
        self.landmark_mapping = dict([(i, self.landmark_labels[i]) for i in range(26)])
        self.cropped_images_dict = read_dict(self.base_output_folder)

    def init_model(self):
        self.model = self.network(num_labels=1, **self.network_parameters)

    def init_optimizer(self):
        self.learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter,
                                                                                     0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate_schedule, amsgrad=True)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name=self.model.name,
                                                         cv=str(self.cv), additional_info=self.config.info)

    def init_datasets(self):
        dataset_parameters = dict(image_base_folder=self.input_folder,
                                  base_folder=self.base_output_folder,
                                  setup_base_folder=self.base_output_folder,
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
        image = np.expand_dims(generators['image'], axis=0)
        single_heatmap = np.expand_dims(generators['single_heatmap'], axis=0)
        image_heatmap_concat = tf.concat([image, single_heatmap],
                                         axis=1 if self.data_format == 'channels_first' else -1)

        prediction = self.model(image_heatmap_concat, training=False)
        prediction = np.squeeze(prediction, axis=0)
        transformation = transformations['image']
        image = generators['image']
        return image, prediction, transformation

    def test(self):
        print('Testing Vertebrae Segmentation...')
        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        filter_largest_cc = True

        for image_id in tqdm(self.test_id_list, desc='Segmentation'):
            first = True
            prediction_labels_np = None
            prediction_max_value_np = None
            full_image = None

            if image_id not in self.valid_landmarks:
                print(f"Skipping {image_id}, no valid landmarks found.")
                continue

            # 【修改】创建独立文件夹 Results/Filename/step3_vertebrae_segmentation
            sample_dir = os.path.join(self.base_output_folder, image_id, 'step3_vertebrae_segmentation')
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)

            for landmark_id in self.valid_landmarks[image_id]:
                dataset_entry = self.dataset_val.get({'image_id': image_id, 'landmark_id': landmark_id})
                if first:
                    input_image = dataset_entry['datasources']['image']
                    full_image = get_full_image(image_id, self.cropped_images_dict, input_image)
                    prediction_labels_np = np.zeros(list(reversed(full_image.GetSize())), dtype=np.uint8)
                    prediction_max_value_np = np.ones(list(reversed(full_image.GetSize())), dtype=np.float32) * 0.5
                    first = False

                image, prediction, transformation = self.test_full_image(dataset_entry)
                del dataset_entry

                origin = transformation.TransformPoint(np.zeros(3, np.float64))
                max_index = transformation.TransformPoint(
                    np.array(self.image_size, np.float64) * np.array(self.image_spacing, np.float64))

                if self.save_output_images:
                    # 保存到子文件夹
                    utils.io.image.write_multichannel_np(image,
                                                         os.path.join(sample_dir,
                                                                      image_id + '_' + landmark_id + '_input.nii.gz'),
                                                         output_normalization_mode='min_max',
                                                         sitk_image_output_mode='vector', data_format=self.data_format,
                                                         image_type=np.uint8, spacing=self.image_spacing, origin=origin)
                    utils.io.image.write_multichannel_np(prediction,
                                                         os.path.join(sample_dir,
                                                                      image_id + '_' + landmark_id + '_prediction.nii.gz'),
                                                         output_normalization_mode=(0, 1),
                                                         sitk_image_output_mode='vector', data_format=self.data_format,
                                                         image_type=np.uint8, spacing=self.image_spacing, origin=origin)

                prediction = prediction.astype(np.float32)
                prediction_resampled_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction,
                                                                                               output_spacing=self.image_spacing,
                                                                                               channel_axis=channel_axis,
                                                                                               input_image_sitk=full_image,
                                                                                               transform=transformation,
                                                                                               interpolator='cubic',
                                                                                               output_pixel_type=sitk.sitkFloat32)
                del prediction
                prediction_resampled_np = utils.sitk_np.sitk_to_np(prediction_resampled_sitk[0])

                bb_start = np.floor(np.flip(origin / np.array(full_image.GetSpacing())))
                bb_start = np.maximum(bb_start, [0, 0, 0])
                bb_end = np.ceil(np.flip(max_index / np.array(full_image.GetSpacing())))
                bb_end = np.minimum(bb_end, prediction_resampled_np.shape - np.ones(3))
                slices = tuple([slice(int(bb_start[i]), int(bb_end[i] + 1)) for i in range(3)])

                prediction_resampled_cropped_np = prediction_resampled_np[slices]

                if filter_largest_cc:
                    prediction_thresh_cropped_np = (prediction_resampled_cropped_np > 0.5).astype(np.uint8)
                    largest_connected_component = utils.np_image.largest_connected_component(
                        prediction_thresh_cropped_np)
                    prediction_thresh_cropped_np[largest_connected_component == 1] = 0
                    prediction_resampled_cropped_np[prediction_thresh_cropped_np == 1] = 0

                prediction_max_value_cropped_np = prediction_max_value_np[slices]
                prediction_labels_cropped_np = prediction_labels_np[slices]

                prediction_max_index_np = utils.np_image.argmax(
                    np.stack([prediction_max_value_cropped_np, prediction_resampled_cropped_np], axis=-1), axis=-1)
                prediction_max_index_new_np = prediction_max_index_np == 1

                prediction_max_value_cropped_np[prediction_max_index_new_np] = prediction_resampled_cropped_np[
                    prediction_max_index_new_np]
                prediction_labels_cropped_np[prediction_max_index_new_np] = self.landmark_mapping[int(landmark_id)]

                prediction_max_value_np[slices] = prediction_max_value_cropped_np
                prediction_labels_np[slices] = prediction_labels_cropped_np
                del prediction_resampled_sitk

            if full_image is not None:
                prediction_labels = utils.sitk_np.np_to_sitk(prediction_labels_np)
                prediction_labels.CopyInformation(full_image)
                # 【修改】保存最终结果到子文件夹
                utils.io.image.write(prediction_labels, os.path.join(sample_dir, image_id + '_seg.nii.gz'))
                print(f"Saved segmentation for {image_id}")
            else:
                print(f"No landmarks found for {image_id}, skipping.")


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def run_inference_step3(config_dict, input_folder, output_folder, model_path):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    config = dotdict(config_dict)
    with MainLoop(0, config, input_folder, output_folder) as loop:
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
