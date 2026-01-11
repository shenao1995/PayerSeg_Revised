import argparse
from glob import glob
import os
import numpy as np
import itk
import nibabel as nib
import SimpleITK as sitk
import time
from utils.save_read_dict import save_dict

# 命令行参数解析（保留以备独立运行，但在被 main_test 调用时不会用到）
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='E:/pythonWorkplace/payer_spine_seg/dataset/test_images')
parser.add_argument('--output_folder', type=str,
                    default='E:/pythonWorkplace/payer_spine_seg/dataset/test_images_reoriented')
parser.add_argument('--sigma', type=float, default=0.75)
args, _ = parser.parse_known_args()  # 使用 parse_known_args 防止与 main_test 参数冲突


def image_cropped(filenames, delete_original_image=False, overlap_size=0):
    cropped_images_dict = {}
    for filename in filenames:
        basename = os.path.basename(filename)
        basename_wo_ext = basename[:basename.find('.nii.gz')]
        if not basename_wo_ext.endswith("_seg"):
            file_path = os.path.dirname(filename)
            try:
                image_total = sitk.ReadImage(filename)
                size = image_total.GetSize()
                # Z轴过长切分逻辑
                if size[2] > 3000:
                    print(f"[{basename_wo_ext}] Z-axis too large ({size[2]}), cropping...")
                    if size[2] % 2 == 1:
                        image_top = image_total[:, :, int(0.5 * size[2]) - overlap_size:]
                        image_bottom = image_total[:, :, :int(0.5 * size[2]) + overlap_size + 1]
                    else:
                        image_top = image_total[:, :, int(0.5 * size[2]) - overlap_size:]
                        image_bottom = image_total[:, :, :int(0.5 * size[2]) + overlap_size]

                    sitk.WriteImage(image_top, os.path.join(file_path, basename_wo_ext + "_top.nii.gz"))
                    sitk.WriteImage(image_bottom, os.path.join(file_path, basename_wo_ext + "_bottom.nii.gz"))

                    with open(os.path.join(file_path, "./test_nii_name.txt"), "a") as f:
                        f.write(basename_wo_ext + "_top" + "\r\n")
                        f.write(basename_wo_ext + "_bottom" + "\r\n")

                    if delete_original_image:
                        os.remove(filename)
                    cropped_images_dict[basename_wo_ext] = "True"
                else:
                    cropped_images_dict[basename_wo_ext] = "False"
                    with open(os.path.join(file_path, "./test_nii_name.txt"), "a") as f:
                        f.write(basename_wo_ext + "\r\n")
            except Exception as e:
                print(f"Error processing cropping for {filename}: {e}")

    return cropped_images_dict


def RAS_or_LPS(nii_path):
    """
    根据 Nibabel 读取的仿射矩阵判断原始方向。
    """
    try:
        image = nib.load(nii_path)
        axcodes = tuple(nib.aff2axcodes(image.affine))
        # print(f"[{os.path.basename(nii_path)}] Orientation: {axcodes}")

        if axcodes == ('R', 'A', 'S'):
            # RAS 数据通常需要翻转矩阵以适配 ITK 默认的 RAI
            return np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64)

        # 默认返回单位矩阵 (适配 LPS 等)
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)
    except Exception:
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64)


def reorient_to_rai(image, filename):
    """
    基于文件名的动态重定向。
    """
    filter = itk.OrientImageFilter[type(image), type(image)].New()
    filter.UseImageDirectionOn()
    filter.SetInput(image)

    # 【修复】：使用动态矩阵
    matrix_np = RAS_or_LPS(filename)
    m = itk.GetMatrixFromArray(matrix_np)

    filter.SetDesiredCoordinateDirection(m)
    filter.Update()
    return filter.GetOutput()


def smooth(image, sigma):
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.SmoothingRecursiveGaussianImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetSigma(sigma)
    filter.Update()
    smoothed = filter.GetOutput()
    return smoothed


def clamp(image):
    ImageType = itk.Image[itk.SS, 3]
    filter = itk.ClampImageFilter[ImageType, ImageType].New()
    filter.SetInput(image)
    filter.SetBounds(-1024, 8192)  # 截断 HU 值
    filter.Update()
    clamped = filter.GetOutput()
    return clamped


def process_image_nodata(filename, output_folder, sigma):
    basename = os.path.basename(filename)
    basename_wo_ext = basename[:basename.find('.nii.gz')]

    ImageType = itk.Image[itk.SS, 3]
    reader = itk.ImageFileReader[ImageType].New()
    # 规范化路径
    norm_filename = os.path.normpath(filename)
    reader.SetFileName(norm_filename)

    try:
        reader.Update()
    except RuntimeError as e:
        print(f"Error reading file: {norm_filename}")
        print(e)
        return
    image = reader.GetOutput()

    print(f'正在处理: {basename_wo_ext} -> reorienting...')

    # 1. 重定向 (Reorient) - 传入 filename 以进行动态检测
    reoriented = reorient_to_rai(image, filename)

    # 2. 强制重置物理坐标系 (Reset Origin/Direction)
    reoriented.SetOrigin([0, 0, 0])
    m = itk.GetMatrixFromArray(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
    reoriented.SetDirection(m)
    reoriented.Update()

    # 3. 【核心修复】预处理 (Smooth & Clamp) - 必须取消注释！
    if not basename_wo_ext.endswith('_seg'):
        reoriented = smooth(reoriented, sigma)
        reoriented = clamp(reoriented)

    # 保存结果
    output_path = os.path.join(output_folder, basename_wo_ext + '.nii.gz')
    itk.imwrite(reoriented, output_path)

    # 可选：打印一下最终确认的方向（调试用）
    # RAS_or_LPS(output_path)


def roi_before(input_folder, output_folder, sigma=0.75):
    """
    修改后的预处理入口，被 main_test_overlap_cropped.py 调用
    """
    roi_time_start = time.time()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 清理旧的记录文件，防止追加重复内容
    test_txt_path = os.path.join(output_folder, "test_nii_name.txt")
    if os.path.exists(test_txt_path):
        os.remove(test_txt_path)

    filenames = glob(os.path.join(input_folder, '*.nii.gz'))
    print(f"[Preprocess] 共发现 {len(filenames)} 个文件")

    # 单线程处理循环
    for filename in sorted(filenames):
        try:
            # 传入参数 output_folder
            process_image_nodata(filename, output_folder, sigma)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # 裁剪逻辑 (如果图像过大)
    filenames_oriented = glob(os.path.join(output_folder, '*.nii.gz'))
    cropped_image_dict = image_cropped(filenames_oriented, delete_original_image=False, overlap_size=128)

    save_dict(cropped_image_dict, output_folder)
    print(f"[Preprocess] Time: {time.time() - roi_time_start:.2f} s")


if __name__ == '__main__':
    # 允许独立运行测试
    if args.image_folder and args.output_folder:
        roi_before(args.image_folder, args.output_folder, args.sigma)