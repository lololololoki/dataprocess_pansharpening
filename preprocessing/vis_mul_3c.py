import os
import numpy as np
from PIL import Image


def save_mul_pic(img, dst_path):
    """
    save mul pic with pre 3 channel
    :param img:
    :param dst_path:
    :return:
    """
    img = img[:, :, 0:3]
    image = Image.fromarray(img)
    image.save(dst_path)
    # img = img[:, :, (2, 1, 0)]
    # image = Image.fromarray(img)
    # image.save(dst_path.replace('.jpg', '_1.jpg'))
    pass


def main():
    pic_path = '/data/jky/jky/tool/dataprocess_pansharpening/VOCdevkit/txt/0_mul.tif'
    image = Image.open(pic_path)
    image = np.asarray(image)
    save_mul_pic(image, pic_path.replace('.tif', '.jpg'))
    pass


if __name__ == '__main__':
    main()
