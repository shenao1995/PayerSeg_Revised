
from datasources.image_datasource import ImageDataSource
from cachetools import LRUCache
import re
from threading import Lock


class LRUCacheWithMissingFunction(LRUCache):
    def __init__(self, maxsize, getsizeof=None, missing=None):
        super(LRUCacheWithMissingFunction, self).__init__(maxsize, getsizeof)
        self.missing = missing

    def __missing__(self, key):
        value = self.missing(key)
        try:
            self.__setitem__(key, value)
        except ValueError:
            pass  # value too large
        return value


class CachedImageDataSource(ImageDataSource):
    """
    # 加载sitk图像
    DataSource used for loading sitk images. Uses id_dict['image_id'] as image path and returns the sitk_image at the given path.
    Supports caching for holding the images in memory.
    Preprocesses the path as follows: file_path_to_load = os.path.join(root_location, file_prefix + id_dict['image_id'] + file_suffix + file_ext)
    FIXME: has some problems when doing cross validation, i.e., memory is sometimes not freed.
    """
    def __init__(self,
                 root_location,
                 file_prefix='',
                 file_suffix='',
                 file_ext='.mha',
                 cache_maxsize=8192,
                 *args, **kwargs):
        """
        Initializer.
        :param root_location: Root path, where the images will be loaded from.
        :param file_prefix: Prefix of the file path.
        :param file_suffix: Suffix of the file path.
        :param file_ext: Extension of the file path.
        :param cache_maxsize: Max size of cache in MB.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(CachedImageDataSource, self).__init__(root_location=root_location,
                                                    file_prefix=file_prefix,
                                                    file_suffix=file_suffix,
                                                    file_ext=file_ext,
                                                    *args, **kwargs)
        self.cache = LRUCacheWithMissingFunction(cache_maxsize, getsizeof=self.image_size, missing=self.load_and_preprocess)
        self.lock = Lock()

    def clear_cache(self):
        """
        Clears the cache.
        """
        with self.lock:
            self.cache.clear()

    def image_size(self, image):
        """
        Returns the image size in MB. Used for calculating the current cache size.
        :param image: The sitk image or a list of sitk images.
        :return: The size of the image in MB.
        """
        # it could be the case that the image is None (if self.return_none_if_not_found == True)
        if image is None:
            return 0
        reference_image = image
        scale_factor = 1
        if isinstance(image, list) or isinstance(image, tuple):
            reference_image = image[0]
            scale_factor = len(image)
        try:
            # ugly silent catch, but the calculated size is only an estimate and we do not care
            """
            这里使用了正则表达式获取每个像素占用的比特数目,其中reference_image.GetPixelIDTypeAsString()一般会返回类似ITK.float32
            这样re.seach就会返回第一个连续数字,使用[0]是因为正则表达式一般获取的是一个列表，取[0]才能获得元素，这样就得到了比特数目
            for example:
            reference_image.GetPixelIDTypeAsString() = float32
            int(re.search('\d+', reference_image.GetPixelIDTypeAsString())[0]) = 32
            """
            num_bits_per_pixel = int(re.search('\d+', reference_image.GetPixelIDTypeAsString())[0])
        except:
            # fallback, if something went wrong (regular expression, unknown pixel id string)
            num_bits_per_pixel = 8
        """
        这里reference_image.GetNumberOfPixels()返回每个图像单独一个通道的像素数目
        reference_image.GetNumberOfComponentsPerPixel()返回通道总数
        """
        total_num_bits = reference_image.GetNumberOfPixels() * reference_image.GetNumberOfComponentsPerPixel() * num_bits_per_pixel * scale_factor
        return total_num_bits / 8 / 1024 / 1024

    def get(self, id_dict):
        """
        Returns the cached image for a given id_dict. If the image is not in the cache, loads and processes the image for the given id_dict. Returns the sitk image.
        :param id_dict: The id_dict. id_dict['image_id'] will be used as the path for loading the sitk image.
        :return: The loaded and processed sitk image.
        """
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict['image_id']
        with self.lock:
            return self.cache[image_id]
