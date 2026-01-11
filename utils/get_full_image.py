import SimpleITK as sitk
import utils.io.image 
import numpy as np
import os
from transformations.intensity.sitk.shift_scale_clamp import ShiftScaleClamp as ShiftScaleClampSitk
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk


class dotdict(dict):
    """
    Dict subclass that allows dot.notation to access attributes.
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def path_for_id(full_image_id):
    current_path = os.getcwd()
    full_image_path = os.path.abspath(os.path.join(current_path, "..", "verse2020_dataset/test_images_reoriented"))
    
    return os.path.join(full_image_path, "" + full_image_id + "" + ".nii.gz")

def load_image(path):
    return utils.io.image.read(path, sitk_pixel_type=sitk.sitkInt16)

config = dotdict(set_zero_origin=False,
                set_identity_direction=False,
                set_identity_spacing=False,
                sitk_pixel_type=sitk.sitkInt16,
                round_spacing_precision=None,
                input_gaussian_sigma=0.75
                )

def intensity_preprocessing_ct(image, input_gaussian_sigma):
        """
        在重采样前,对加载的sitk图像进行高斯处理
        Intensity preprocessing function, working on the loaded sitk image, before resampling.
        :param image: The sitk image.
        :return: The preprocessed sitk image.
        """
        image = ShiftScaleClampSitk(clamp_min=-1024)(image)
        if input_gaussian_sigma > 0:
            return gaussian_sitk(image, input_gaussian_sigma)
        return image

def preprocess(config, image):
        if image is None:
            return image
        if config.set_identity_spacing:
            image.SetSpacing([1] * image.GetDimension())
        if config.set_zero_origin:
            image.SetOrigin([0] * image.GetDimension())
        if config.set_identity_direction:
            image.SetDirection(np.eye(image.GetDimension()).flatten())
        if config.round_spacing_precision is not None:
            image.SetSpacing([round(x, config.round_spacing_precision) for x in image.GetSpacing()])
        image = intensity_preprocessing_ct(image, config.input_gaussian_sigma)
        return image

def load_and_preprocess(full_image_id):
    image = load_image(path_for_id(full_image_id))
    image = preprocess(config, image)
    return image


def get_full_image(current_id, cropped_images_dict, input_image):
    if current_id.endswith("_top"):
        new_current_id = current_id.replace("_top", "")
        if new_current_id in list(cropped_images_dict.keys()) and cropped_images_dict[new_current_id] == "True":
            full_before_cropped_image = load_and_preprocess(new_current_id)
        else:
            full_before_cropped_image = input_image
    if current_id.endswith("_bottom"):
        new_current_id = current_id.replace("_bottom", "")
        if new_current_id in list(cropped_images_dict.keys()) and cropped_images_dict[new_current_id] == "True":
            full_before_cropped_image = load_and_preprocess(new_current_id)
        else:
            full_before_cropped_image = input_image
    if not current_id.endswith("_top") and not current_id.endswith("_bottom"):
        full_before_cropped_image = input_image
    return full_before_cropped_image