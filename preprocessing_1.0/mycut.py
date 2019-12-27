# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import gdal,ogr,osr
import sys
import shutil
from genTif8bit import gen_pan, gen_mul
import random
from TIFF2JPG import panTif2Gray

"""
    transform shp to txt
    divide tif into 512x512 or other size
"""

classDefn={'厂房':1,'砖木':2,'砖混':3,'配房':4,'钢混':5,'其它':6,'其他':6, '住宅':6, '住房':7, '其他1':8, '其他2':9, '大棚':10, '其它2':11}
def shp2txt(srcFile, shpFile, outputFile):
    gdal.SetConfigOption("SHAPE_ENCODING","")
    dataset = gdal.Open(srcFile)
    boxFile = "%s_boxes" % outputFile
    pointFile = "%s_points" % outputFile
    boxOutput = open(boxFile, "wb")
    pointOutput = open(pointFile, "wb")

    xSize = dataset.RasterXSize
    ySize = dataset.RasterYSize

    print("x:%d,y:%d" % (xSize, ySize))

    geoTransform = dataset.GetGeoTransform()

    dataSource = ogr.Open(shpFile)

    print(dataSource)
    
    daLayer = dataSource.GetLayer(0)

    featureCount = daLayer.GetFeatureCount()
    layerDefinition = daLayer.GetLayerDefn()
    
    print(layerDefinition.GetFieldCount())
    
    for i in range(layerDefinition.GetFieldCount()):
        # fieldName = layerDefinition.GetFieldDefn(i).GetName().decode("gb2312")
        fieldName = layerDefinition.GetFieldDefn(i).GetName().decode("utf8")
        print(fieldName)
        fieldTypeCode = layerDefinition.GetFieldDefn(i).GetType()
        fieldType = layerDefinition.GetFieldDefn(i).GetFieldTypeName(fieldTypeCode)
        if fieldType=="String" and fieldName != u"房屋用途":
            name=fieldName
        fieldWidth = layerDefinition.GetFieldDefn(i).GetWidth()
        GetPrecision = layerDefinition.GetFieldDefn(i).GetPrecision()
    print(featureCount)
    for j in range(featureCount):
        feature = daLayer.GetFeature(j)
        ###jky###
        #print featureCount
        #print feature
        #print feature.GetField(name.encode("gb2312"))
        #print feature.GetField(fieldName.encode("gb2312"))
        #print name
        #print fieldName
        # className = feature.GetField(name.encode("gb2312")).decode('gb2312').encode('utf8')
        # print(featureCount)
        # className = feature.GetField(name.encode("gb2312")).decode('gb2312').encode('utf8')
        className = feature.GetField(name.encode("utf8"))
        # print("className:", className)
        # print("classDefn:", classDefn)
        if className not in classDefn:
            print "skip",className
            print srcFile
            # continue
        if className is not None:
            with open('className_temp.txt', 'w') as f:
                f.write(className)
            # print('className:', className.decode('utf-8'))
            buildingType=classDefn[className]
        else:
            buildingType=9
        geometry = feature.GetGeometryRef()
        # print(geometry)
        if geometry == None:
            continue
        ring = geometry.GetGeometryRef(0)
        numPoints = ring.GetPointCount()

        maxCol = 0
        minCol = sys.maxsize
        maxRow = 0
        minRow = sys.maxsize
        #pointOutput.write("%d " % numPoints)

        points = []
        
        for i in range(numPoints):
            
            #print "numPoints:"
            #print numPoints
            #print "i:"
            #print i
            if(numPoints == i+1):
                #print i
                break
            
            dTemp = geoTransform[1] * geoTransform[5] - \
                geoTransform[2] * geoTransform[4]
            dcol = 0.0
            drow = 0.0
            dcol = (geoTransform[5] * (ring.GetX(i) - geoTransform[0]) -
                    geoTransform[2] * (ring.GetY(i) - geoTransform[3])) / dTemp + 0.5
            drow = (geoTransform[1] * (ring.GetY(i) - geoTransform[3]) -
                    geoTransform[4] * (ring.GetX(i) - geoTransform[0])) / dTemp + 0.5
            icol = int(dcol)
            irow = int(drow)
            #pointOutput.write("%d %d " % (icol, irow))
            points.append(icol)
            points.append(irow)
            if(icol > maxCol):
                maxCol = icol
            if(irow > maxRow):
                maxRow = irow
            if(icol < minCol):
                minCol = icol
            if(irow < minRow):
                minRow = irow

        #pointOutput.write("%d\n" % numPoints)
        #pointOutput.write("\n")
        # print("%d:%d,%d,%d,%d\n" % (j,minCol, minRow, maxCol, maxRow))
        if numPoints != 0:
            if not (maxCol-minCol<30 & maxRow-minRow<30):
                boxOutput.write("%d,%d,%d,%d,%d\n" % (minCol, minRow, maxCol, maxRow,buildingType))
                for i in range(len(points)):    
                    pointOutput.write("%d " % points[i])
                pointOutput.write("\n")
    pointOutput.close()
    boxOutput.close()
    print "done!"

# lxy
def batShp2Txt_lxy(filelist, outDir):
    files = open(filelist).readlines()
    for (line,i) in zip(files,range(len(files))):
        srcfile,shpfile=line.split()
        print srcfile,shpfile
        shp2txt(srcfile, shpfile, '%s/%d' % (outDir, i))
        shutil.copyfile(srcfile, '%s/%d.tif' % (outDir, i))

def batShp2Txt(filelist, outDir):
    mul_path = "../多光谱/"
    files = open(filelist).readlines()
    for (line,i) in zip(files,range(len(files))):
        srcfile,shpfile=line.split()
        print srcfile,shpfile
        shp2txt(srcfile, shpfile, '%s/%d' % (outDir, i))
        gen_pan(srcfile, '%s/%d.tif' % (outDir, i))
        _, fullflname = os.path.split(srcfile)
        if os.path.exists(os.path.join(mul_path, fullflname)):
            # TODO wrinting gen_mul
            _, fullflname = os.path.split(srcfile)
            gen_mul(os.path.join(mul_path, fullflname), '%s/%d_mul.tif' % (outDir, i))
        # shutil.copyfile(srcfile, '%s/%d.tif' % (outDir, i))

def division_one_tif(inDir,outDir,i,count):
    count_old=count
    print "now:",i
    boxFile="%s/%d_boxes"%(inDir,i)
    lines=open(boxFile).readlines()
    boxes=[]
    
    for line in lines:
        boxes.append([int(item) for item in line.split(",")])
        
    ###jky###
    pointFile="%s/%d_points"%(inDir,i)
    points_lines=open(pointFile)
    points=[]
    
    for line in points_lines:
        points.append([int(item) for item in line.split()])
    
    #print "points:"
    #print points[0]
    #print len(points[0])
    
    PIECE_X=512 ###jky### 1280
    PIECE_Y=512 ###jky### 960
    dataset=gdal.Open('%s/%d.tif'%(inDir,i))
    dataset_mul=gdal.Open('%s/%d_mul.tif'%(inDir,i))
    xSize=dataset.RasterXSize
    ySize=dataset.RasterYSize
    print(xSize, ySize)
    for x in range(0,xSize-PIECE_X,PIECE_X):
        for y in range(0,ySize-PIECE_Y,PIECE_Y):
            x1=x
            y1=y
            x2=x+PIECE_X
            y2=y+PIECE_Y
            boxes_in_block=[]
            points_in_block=[]
            #for box in boxes:
            #    boxHandler(box,[x1,y1,x2,y2],boxes_in_block)
            
            for (box,point) in zip(boxes,points):
                #box = boxes[i]
                #point = points[i]
                #print box
                #print point
                pointHandler(box,point,[x1,y1,x2,y2],boxes_in_block,points_in_block)
                #print [x1,y1,x2,y2]
                #print boxes_in_block
                #print points_in_block
            
            print(len(boxes_in_block))
            if len(boxes_in_block)>5:
                genNewTif(dataset,"%s/JPEGImages/%06d.jpg"%(outDir,count),x1,y1,PIECE_X,PIECE_Y)
                genNewTif_pan(dataset,"%s/pan/%06d_pan.tif"%(outDir,count),x1,y1,PIECE_X,PIECE_Y)
                genNewTif_mul(dataset_mul, "%s/mul/%06d_mul.tif" % (outDir, count), x1/4, y1/4, PIECE_X/4, PIECE_Y/4)
                box2txt(boxes_in_block,"%s/boxes/%06d.txt"%(outDir,count))
                point2txt(points_in_block,"%s/points/%06d.txt"%(outDir,count))
                count+=1
    sum=count-count_old
    #if sum<100:
    #    for temp in range(100-sum):
    #        x1=random.randint(0,xSize-PIECE_X)
    #       y1=random.randint(0,ySize-PIECE_Y)
    #        x2 = x1 + PIECE_X
    #        y2 = y1 + PIECE_Y
    #        boxes_in_block = []
    #        for box in boxes:
    #            boxHandler(box, [x1, y1, x2, y2], boxes_in_block)
    #        if len(boxes_in_block) > 5:
    #            genNewTif(dataset, "%s/JPEGImages/%06d.jpg" % (outDir, count), x1, y1, PIECE_X, PIECE_Y)
    #            box2txt(boxes_in_block, "%s/Annotations/%06d.txt" % (outDir, count))
    #            count += 1
    return count

def box2txt(boxes, filename):
    output = open(filename, "wb")
    for box in boxes:
        line = "%d %d %d %d %d\n" % (box[0], box[1], box[2], box[3], box[4])
        output.write(line)
    output.close()

def point2txt(points, filename):
    output = open(filename, "wb")
    for point in points:
        line = ""
        for dot in point:
            line = line + "%d " % dot  
        line = line + "\n"
        output.write(line)
    output.close()

def linear(img):
    img_new=np.zeros(img.shape)
    sum_=img.shape[0]*img.shape[1]
    print sum_
    
    num=np.zeros(5000)
    prob=np.zeros(5000)
    for j in range(0,img.shape[0]):
        for k in range(0,img.shape[1]):
            num[img[j,k]]=num[img[j,k]]+1
    for tmp in range(0,5000):
        prob[tmp]=num[tmp]/sum_
    Min=0
    Max=0
    min_prob=0.0 
    max_prob=0.0
    while(Min<5000 and min_prob<0.2):
        min_prob+=prob[Min]
        Min+=1
    print min_prob,Min
    while (True):
        max_prob+=prob[Max]
        Max+=1
        if(Max>=5000 or max_prob>=0.98):
            break
    print max_prob,Max
    for m in range(0,img.shape[0]):
        for n in range(0,img.shape[1]):
            if (img[m,n]>Max):
                img_new[m,n]=255
            elif(img[m,n]<Min):
                img_new[m,n]=0
            else:
                img_new[m,n]=(img[m,n]-Min)/(Max-Min)*255
    return img_new

def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):

    cols = array.shape[1]
    rows = array.shape[0]
    originX = rasterOrigin[0]
    originY = rasterOrigin[1]

    driver = gdal.GetDriverByName('GTiff')
    
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    for i in range(1,2):

        outband = outRaster.GetRasterBand(i)
        outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()


def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, bandSize):
    if (bandSize == 4):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_Byte)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif (bandSize == 1):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array)

# lxy
def genNewTif(raster, dstFile, x, y, cols, rows):
    # temp=raster.ReadAsArray(x,y,cols,rows).transpose([1,2,0])
    # temp=temp[:,:,::-1]
    temp=raster.ReadAsArray(x,y,cols,rows)
    print(dstFile)
    cv2.imwrite('%s' % dstFile, temp)

def genNewTif_pan(raster, dstFile, x, y, cols, rows):
    temp=raster.ReadAsArray(x,y,cols,rows)
    print(dstFile)
    array2raster(dstFile, (-123.25745,45.43013), 2.4, 2.4, temp, 1)

def genNewTif_mul(raster, dstFile, x, y, cols, rows):
    temp=raster.ReadAsArray(x,y,cols,rows)
    print(dstFile)
    array2raster(dstFile, (-123.25745,45.43013), 2.4, 2.4, temp, 4)
    
def genNewTif_1(raster, dstFile, x, y, cols, rows):
    # temp=raster.ReadAsArray(x,y,cols,rows).transpose([1,2,0])
    # temp=temp[:,:,::-1]
    rasterOrigin = (-123.25745,45.43013)
    temp=raster.ReadAsArray(x,y,cols,rows)
    temp=linear(temp)
    print(dstFile)
    # cv2.imwrite('%s' % dstFile, temp)

def boxHandler(box, block, boxes_in_block):
    box_new = box[::]
    if ((box[0] >= block[0] and box[1] >= block[1] and box[0] < block[2] and box[1] < block[3]) and (
            box[2] >= block[0] and box[3] >= block[1] and box[2] < block[2] and box[3] < block[3])):
        boxes_in_block.append([box_new[0] -
                               block[0], box_new[1] -
                               block[1], box_new[2] -
                               block[0], box_new[3] -
                               block[1],box_new[4]])
    elif (box[0] >= block[0] and box[1] >= block[1] and box[0] < block[2] and box[1] < block[3]):
        if box[2] >= block[2]:
            box_new[2] = block[2] - 1
        if box[3] >= block[3]:
            box_new[3] = block[3] - 1
        s = (box[2] - box[0]) * (box[3] - box[1])
        s_new = (box_new[2] - box_new[0]) * (box_new[3] - box_new[1])
        # print s,s_new
        if (float(s_new) / s >= 0.5):
            #print float(s_new) / s
            boxes_in_block.append([box_new[0] -
                                   block[0], box_new[1] -
                                   block[1], box_new[2] -
                                   block[0], box_new[3] -
                                   block[1],box_new[4]])
    elif (box[2] >= block[0] and box[3] >= block[1] and box[2] < block[2] and box[3] < block[3]):
        if box[0] < block[0]:
            box_new[0] = block[0]
        if box[1] < block[1]:
            box_new[1] = block[1]
        s = (box[2] - box[0]) * (box[3] - box[1])
        s_new = (box_new[2] - box_new[0]) * (box_new[3] - box_new[1])
        # print s,s_new
        if (float(s_new) / s >= 0.5):
            boxes_in_block.append([box_new[0] -
                                   block[0], box_new[1] -
                                   block[1], box_new[2] -
                                   block[0], box_new[3] -
                                   block[1],box_new[4]])

def pointHandler(box, point, block, boxes_in_block, points_in_block):
    halflen_point = len(point)/2
    #print halflen_point
    box_new = box[::]
    point_new = point[::]
    #print point
    if ((box[0] >= block[0] and box[1] >= block[1] and box[0] < block[2] and box[1] < block[3]) and (
            box[2] >= block[0] and box[3] >= block[1] and box[2] < block[2] and box[3] < block[3])):
        boxes_in_block.append([box_new[0] -
                               block[0], box_new[1] -
                               block[1], box_new[2] -
                               block[0], box_new[3] -
                               block[1],box_new[4]])
        point_in_block = []
        for i in range(len(point_new)):
            #偶数
            if (i % 2) == 0:
                point_in_block.append(point_new[i] - block[0])
            #奇数
            else:
                point_in_block.append(point_new[i] - block[1])
        points_in_block.append(point_in_block)
    elif (box[0] >= block[0] and box[1] >= block[1] and box[0] < block[2] and box[1] < block[3]):
        #xmax >= 
        if box[2] >= block[2]:
            box_new[2] = block[2] - 1
            for i in range(halflen_point):
                if point[2*i] >= block[2]:
                    point_new[2*i] = block[2] - 1
        #ymax >=
        if box[3] >= block[3]:
            box_new[3] = block[3] - 1
            for i in range(halflen_point):
                if point[2*i+1] >= block[3]:
                    point_new[2*i+1] = block[3] - 1
        s = (box[2] - box[0]) * (box[3] - box[1])
        s_new = (box_new[2] - box_new[0]) * (box_new[3] - box_new[1])
        # print s,s_new
        if (float(s_new) / s >= 0.5):
            #print float(s_new) / s
            boxes_in_block.append([box_new[0] -
                                   block[0], box_new[1] -
                                   block[1], box_new[2] -
                                   block[0], box_new[3] -
                                   block[1],box_new[4]])
            point_in_block = []
            for i in range(len(point_new)):
                #偶数
                if (i % 2) == 0:
                    point_in_block.append(point_new[i] - block[0])
                #奇数
                else:
                    point_in_block.append(point_new[i] - block[1])
            points_in_block.append(point_in_block)
    elif (box[2] >= block[0] and box[3] >= block[1] and box[2] < block[2] and box[3] < block[3]):
        #xmin <
        if box[0] < block[0]:
            box_new[0] = block[0]
            for i in range(halflen_point):
                if point[2*i] < block[0]:
                    point_new[2*i] = block[0]
        #ymin <
        if box[1] < block[1]:
            box_new[1] = block[1]
            for i in range(halflen_point):
                if point[2*i+1] < block[1]:
                    point_new[2*i+1] = block[1]
        s = (box[2] - box[0]) * (box[3] - box[1])
        s_new = (box_new[2] - box_new[0]) * (box_new[3] - box_new[1])
        # print s,s_new
        if (float(s_new) / s >= 0.5):
            boxes_in_block.append([box_new[0] -
                                   block[0], box_new[1] -
                                   block[1], box_new[2] -
                                   block[0], box_new[3] -
                                   block[1],box_new[4]])
            point_in_block = []
            for i in range(len(point_new)):
                #偶数
                if (i % 2) == 0:
                    point_in_block.append(point_new[i] - block[0])
                #奇数
                else:
                    point_in_block.append(point_new[i] - block[1])
            points_in_block.append(point_in_block)
                                   
def batDivision(inDir,outDir):
    count = 0
    for parent,directory,filenames in os.walk(inDir):
        num=len(filenames)/3
    print('num', num)
    for i in range(num):
        count = division_one_tif(inDir, outDir, i, count)
        print count
    # for i in range(num):
        # try:
            # count = division_one_tif(inDir, outDir, i, count)
            # print count
        # except:
            # print i
        #count = division_one_tif(inDir, outDir, i, count)
        #print count

def visualize(jpg,box,output):
    img=cv2.imread(jpg)
    lines=open(box).readlines()
    lines=[line[0:-1] for line in lines]
    boxes=[[int(item)for item in line.split(' ')[0:4]] for line in lines]
    if box[0]==0 or box[1]==0:
        print "0!!!!"
    for box in boxes:
        cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]),(0,0,255))
    cv2.imwrite(output,img)

def pointvisualize_keypoints(jpg,point,output,keypointsoutput):
    
    keypointsoutput = open(keypointsoutput, "wb")
    keypointsline = ""
    
    img=cv2.imread(jpg)
    lines=open(point).readlines()
    lines=[line[0:-1] for line in lines]
    
    #print lines
    
    points=[[int(item)for item in line.split()] for line in lines]
    
    #print points
    
    if point[0]==0 or point[1]==0:
        print "0!!!!"    
    
    for point in points: 
        #print point
        ploypoints = []
        for i in range(len(point)/2):
            ploypoints.append([point[2*i],point[2*i+1]])
        ploypoints = np.array([ploypoints],dtype = np.int32)
        #原始轮廓
        cv2.polylines(img, ploypoints, 1, (0,255,0))
        cnt = ploypoints
        cnt = cnt.reshape(-1,2)
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.maximum(box,1)
        box = np.minimum(box,511)
        #初始点(最左边)
        cv2.circle(img, tuple(box[0]), 10, (255,0,0), 3)
        ploypoints = box
        ploypoints = np.array([ploypoints],dtype = np.int32)
        #最小外接矩形
        cv2.polylines(img, ploypoints, 1, (0,0,255))

        for dot in ploypoints.tolist()[0]:
            #print (dot[0])
            keypointsline = keypointsline + "%d %d 2 " % (dot[0],dot[1]) 
        keypointsline = keypointsline + "\n"
        
    cv2.imwrite(output,img)
    #keypointsoutput.write(keypointsline)
    #keypointsoutput.close()


def pointvisualize(jpg, point, output, keypointsoutput):
    keypointsoutput = open(keypointsoutput, "wb")
    keypointsline = ""

    img = cv2.imread(jpg)
    lines = open(point).readlines()
    lines = [line[0:-1] for line in lines]

    # print lines

    points = [[int(item) for item in line.split()] for line in lines]

    # print points

    if point[0] == 0 or point[1] == 0:
        print "0!!!!"

    for point in points:
        # print point
        ploypoints = []
        for i in range(len(point) / 2):
            ploypoints.append([point[2 * i], point[2 * i + 1]])
        ploypoints = np.array([ploypoints], dtype=np.int32)
        # 原始轮廓
        cv2.polylines(img, ploypoints, 1, (0, 255, 0))
        cnt = ploypoints
        cnt = cnt.reshape(-1, 2)
        box = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(box)
        box = np.maximum(box, 1)
        box = np.minimum(box, 511)
        # 初始点(最左边)
        cv2.circle(img, tuple(box[0]), 10, (255, 0, 0), 3)
        ploypoints = box
        ploypoints = np.array([ploypoints], dtype=np.int32)
        # 最小外接矩形
        cv2.polylines(img, ploypoints, 1, (0, 0, 255))

        for dot in ploypoints.tolist()[0]:
            # print (dot[0])
            keypointsline = keypointsline + "%d %d " % (dot[0], dot[1])
        keypointsline = keypointsline + "\n"

    #cv2.imwrite(output, img)
    keypointsoutput.write(keypointsline)
    keypointsoutput.close()
    
def batVisualize(jpgs,boxes,output):
    #num=0
    for parent,directory,filenames in os.walk(jpgs):
        #print filenames
        num=len(filenames)
        #print num
    for i in range(0,num,1):
        print i
        visualize('%s/%06d.jpg'%(jpgs,i),'%s/%06d.txt'%(boxes,i),'%s/%06d.jpg'%(output,i))
        
def batpointVisualize(jpgs,points,output,keypointsoutput):
    #num=0
    for parent,directory,filenames in os.walk(jpgs):
        #print filenames
        num=len(filenames)
        #print num
    for i in range(0,num,1):
        print i
        pointvisualize('%s/%06d.jpg'%(jpgs,i),'%s/%06d.txt'%(points,i),'%s/%06d.jpg'%(output,i),'%s/%06d.txt'%(keypointsoutput,i))

#1
batShp2Txt('../data/filelist.txt', '../VOCdevkit/txt')

#2
batDivision('../VOCdevkit/txt', '../VOCdevkit/VOC0712')
