# -*- coding: utf-8 -*-
from __future__ import division
import gdal, ogr, os, osr
import numpy as np
import cv2


def gen_mul(srcFile, dstFile):

    def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_Byte, ['TILED=YES', 'COMPRESS=PACKBITS'])
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()

    def main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
        # reversed_arr = array[::-1] # reverse array so the tif looks like the array
        array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)  # convert array to raster

    def linear(img):
        img_new = np.zeros(img.shape)
        sum_ = img.shape[1] * img.shape[2]
        print sum_
        for i in range(0, img.shape[0]):
            num = np.zeros(5000)
            prob = np.zeros(5000)
            for j in range(0, img.shape[1]):
                for k in range(0, img.shape[2]):
                    num[img[i, j, k]] = num[img[i, j, k]] + 1
            for tmp in range(0, 5000):
                prob[tmp] = num[tmp] / sum_
            Min = 0
            Max = 0
            min_prob = 0.0
            max_prob = 0.0
            while (Min < 5000 and min_prob < 0.2):
                min_prob += prob[Min]
                Min += 1
            print min_prob, Min
            while (True):
                max_prob += prob[Max]
                Max += 1
                if (Max >= 5000 or max_prob >= 0.98):
                    break
            print max_prob, Max
            for m in range(0, img.shape[1]):
                for n in range(0, img.shape[2]):
                    if (img[i, m, n] > Max):
                        img_new[i, m, n] = 255
                    elif (img[i, m, n] < Min):
                        img_new[i, m, n] = 0
                    else:
                        img_new[i, m, n] = (img[i, m, n] - Min) / (Max - Min) * 255
        return img_new

    rasterOrigin = (-123.25745, 45.43013)
    dataset = gdal.Open(srcFile)
    img = dataset.ReadAsArray()
    img = linear(img)
    main(dstFile, rasterOrigin, 2.4, 2.4, img)


def gen_pan(srcFile, dstFile):

    def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 2):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()

    def main(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array):
        # reversed_arr = array[::-1] # reverse array so the tif looks like the array
        array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array)  # convert array to raster

    def linear(img):
        img_new = np.zeros(img.shape)
        sum_ = img.shape[0] * img.shape[1]
        print sum_

        num = np.zeros(5000)
        prob = np.zeros(5000)
        for j in range(0, img.shape[0]):
            for k in range(0, img.shape[1]):
                num[img[j, k]] = num[img[j, k]] + 1
        for tmp in range(0, 5000):
            prob[tmp] = num[tmp] / sum_
        Min = 0
        Max = 0
        min_prob = 0.0
        max_prob = 0.0
        while (Min < 5000 and min_prob < 0.2):
            min_prob += prob[Min]
            Min += 1
        print min_prob, Min
        while (True):
            max_prob += prob[Max]
            Max += 1
            if (Max >= 5000 or max_prob >= 0.98):
                break
        print max_prob, Max
        for m in range(0, img.shape[0]):
            for n in range(0, img.shape[1]):
                if (img[m, n] > Max):
                    img_new[m, n] = 255
                elif (img[m, n] < Min):
                    img_new[m, n] = 0
                else:
                    img_new[m, n] = (img[m, n] - Min) / (Max - Min) * 255
        return img_new

    rasterOrigin = (-123.25745, 45.43013)
    dataset = gdal.Open(srcFile)
    img = dataset.ReadAsArray()
    img = linear(img)
    main(dstFile, rasterOrigin, 2.4, 2.4, img)

if __name__ == "__main__":
    ROOT = '/data/jky/jky/tool/dataprocess_pansharpening/data_86'
    rasterOrigin = (-123.25745, 45.43013)
    for i in range(1,10):
        newMul=os.path.join(ROOT, 'dataset/%d_pan.tif'%i)
        dataset=gdal.Open(os.path.join(ROOT, '%d_PAN.TIF'%i))
        img=dataset.ReadAsArray()
        img=img.transpose(1,0)
        img=cv2.pyrDown(cv2.pyrDown(img))
        img=img.transpose(1,0)
        img=linear(img)
        main(newMul,rasterOrigin,2.4,2.4,img)
        print ('done%d'%i)
    cv2.imwrite('blur_1.tif m',img_blur.transpose(1,2,0))
    tmp=cv2.imread('blur_1.tif',-1)

    # print tmp[234][456]
    # print img_blur.transpose(1,2,0)[234][456]
