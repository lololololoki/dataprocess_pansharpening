#encoding:utf-8
import os
import cv2
import numpy as np
import gdal,ogr,osr
import sys
import shutil
from genTif8bit import gen_pan, gen_mul
import random
from TIFF2JPG import panTif2Gray

dataset = gdal.Open("/data/jky/jky/tool/dataprocess_pansharpening/preprocessing/test.tif")
img = dataset.ReadAsArray()

print(img.max)

mul_path = "/data/jky/jky/tool/dataprocess_pansharpening/data/多光谱/1_MUL.TIF"
gen_mul(mul_path, "/data/jky/jky/tool/dataprocess_pansharpening/data/多光谱/test.tif")