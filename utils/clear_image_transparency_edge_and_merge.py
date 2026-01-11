from PIL import Image
import numpy as np


def clear_image_transparency_edge(image_path):
    """
    根据图片路径裁剪透明边缘
    """
    image = Image.open(image_path)
    image_array = np.array(image)
    height, width, _ = image_array.shape
    # 计算非透明像素的行
    rows = np.where(image_array[:, :, 3] > 0)[0]
    # 计算非透明像素的列
    cols = np.where(image_array[:, :, 3] > 0)[1]
    # 得到非透明元素上下左右第一个索引
    top = rows.min()
    bottom = rows.max()
    left = cols.min()
    right = cols.max()
    image = image.crop((left, top, right + 1, bottom + 1))
    image.save(image_path)


def get_new_path(image_top_path, image_bottom_path):
    new_filename_top = image_top_path.split("/")[-1].replace("_top", "")
    new_path_top = image_top_path.rsplit("/", 1)[0] + "/" + new_filename_top
    new_filename_bottom = image_bottom_path.split("/")[-1].replace("_bottom", "")
    new_path_bottom = image_bottom_path.rsplit("/", 1)[0] + "/" + new_filename_bottom
    if new_path_top == new_path_bottom:
         return new_path_top
    else:
         return "please check your ct name"

def connect_top_and_bottom_image(image_top_path, image_bottom_path, flag = 'y'):
    """
    params image_top_path this is path of  top of image
    params image_bottom_path this is path of bottom of image
    params flag if flag = "y" Longitudinal splicing else Horizontal stitching
    """
    image_top, image_bottom = Image.open(image_top_path), Image.open(image_bottom_path)
    image_top_size, image_bottom_size = image_top.size, image_bottom.size
    if flag == 'x':
          connect_image = Image.new("RGBA", (image_top_size[0] + image_bottom_size[0], image_top_size[1]))
          top_loc, bottom_loc = (0, 0), (image_top_size[0], 0)
    else:
          connect_image = Image.new("RGBA", (image_top_size[0], image_top_size[1] + image_bottom_size[1]))
          top_loc, bottom_loc = (0, 0), (0, image_top_size[1])
    connect_image.paste(image_top, top_loc)
    connect_image.paste(image_bottom, bottom_loc)
    same_path = get_new_path(image_top_path, image_bottom_path)
    connect_image.save(same_path)


def remove_edge_and_connect(image_top_path, image_bottom_path, flag = "y"):
    """
    输入图片分别裁剪透明边缘
    然后合并图片
    """
    clear_image_transparency_edge(image_top_path)
    clear_image_transparency_edge(image_bottom_path)
    connect_top_and_bottom_image(image_top_path, image_bottom_path, flag)


def merge_top_and_bottom_image(image_top_path, image_bottom_path):
    image_top = Image.open(image_top_path)
    image_bottom = Image.open(image_bottom_path)
    image_top = image_top.convert('RGBA')
    image_bottom = image_bottom.convert('RGBA')

    # 确保两张图片大小一致
    if image_top.size != image_bottom.size:
        # 如果不一致，就将第二张图片缩放到和第一张图片大小一致
        image_bottom = image_bottom.resize(image_top.size)
        
    # 将第一张图片的标志点设置为不透明
    alpha_channel = Image.new('L', image_top.size, 255)
    image_top.putalpha(alpha_channel)

    # 将两张图片混合在一起
    merged_image = Image.blend(image_top, image_bottom, 0.5)

    # 保存融合后的图像
    merge_image_path = get_new_path(image_top_path, image_bottom_path)
    merged_image.save(merge_image_path)


def remove_edge_and_merge(image_top_path, image_bottom_path):
    clear_image_transparency_edge(image_top_path)
    clear_image_transparency_edge(image_bottom_path)
    merge_top_and_bottom_image(image_top_path, image_bottom_path)


if __name__ == "__main__":
    image_path1 = "./dai_cai_fang_top_landmarks_pp.png"
    image_path2 = "./dai_cai_fang_bottom_landmarks_pp.png"
    remove_edge_and_merge(image_path1, image_path2)
    current_id = "dai_cai_fang_bottom"
    print(current_id.endswith("_bottom"))
    print(current_id.replace("_bottom", ""))
    print(current_id)
    for a, b in zip([1, 2, 3], [4, 5, 6]):
         print(a, b)
    
    