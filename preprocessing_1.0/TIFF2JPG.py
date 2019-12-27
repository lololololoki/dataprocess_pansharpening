# -*- coding: utf-8 -*-
# from PIL import Image
# im = Image.open('jinan.tif')
# im.save('jinan.jpg')

# import arcpy
# mxd = arcpy.mapping.MapDocument(r"C:\Project\Project.mxd")
# arcpy.mapping.ExportToJPEG(mxd, r"C:\Project\Output\Project.jpg")
# del mxd

import cv2
import gdal
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def panTif2Gray(img, dstFile):
    # img = cv2.imread('jinan.tif', -1)
    plt.set_cmap('gray')
    fig = plt.gcf()
    plt.imshow(img)
    plt.axis('off')
    # 去除图像周围的白边
    height, width = img.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width/100.0, height/100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    fig.savefig(dstFile)
    plt.clf()
    plt.close
    # cv2.imwrite('jinan.jpg', img)

def mulTif2Gray(img, dstFile):
    # img = cv2.imread('jinan.tif', -1)
    # reshape to [H, W ,C]
    img = img.transpose([1,2,0]).mean(2)
    plt.set_cmap('gray')
    fig = plt.gcf()
    plt.imshow(img)
    plt.axis('off')
    # 去除图像周围的白边
    height, width = img.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width/100.0, height/100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    fig.savefig(dstFile)
    # cv2.imwrite('jinan.jpg', img)

def panTif2GrayOpencv(srcFile, dstFile):
    img = cv2.imread(srcFile, 8)
    plt.set_cmap('gray')
    fig = plt.gcf()
    plt.imshow(img)
    plt.axis('off')
    # 去除图像周围的白边
    height, width = img.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width/100.0, height/100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    fig.savefig(dstFile)
    # cv2.imwrite('jinan.jpg', img)

def mulTif2GrayOpencv(srcFile, dstFile):
    img = cv2.imread(srcFile, 8)
    plt.set_cmap('gray')
    fig = plt.gcf()
    plt.imshow(img)
    plt.axis('off')
    # 去除图像周围的白边
    height, width = img.shape
    # 如果dpi=300，那么图像大小=height*width
    fig.set_size_inches(width/100.0, height/100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    fig.savefig(dstFile)
    # cv2.imwrite('jinan.jpg', img)

if __name__ == "__main__":
    srcFile = 'jinan_mul.tif'
    dstFile = 'jinan.jpg'


    dataset=gdal.Open(srcFile)
    img = dataset.ReadAsArray()

    mulTif2GrayOpencv(srcFile, dstFile)
    # panTif2Gray(dataset, dstFile)