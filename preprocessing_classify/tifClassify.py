# -*- coding: utf-8 -*-\
#分类存放 rural suburban urban .tif文件
import os
import shutil

ruralNum = [7, 22, 23, 24, 26, 28, 29, 31, 38, 48]
suburbanNum = [1, 2, 3, 4, 6, 8, 15, 20, 25, 27]
urbanNum = [5, 10, 11, 12, 13, 14, 16, 17, 18, 19]

fileType = ['.tif', '_boxes', '_points']

def tifClassify(srcDir, dstDir):
    r"""
    srcFile: tif文件夹
    """
    os.chdir(srcDir)  
    
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)

    count = 0
        
    # rural
    for num in ruralNum:
        newNum = '0' + str(count)
        count = count + 1
        
        for type in fileType:
            srcFile = srcDir + str(num) + type
            dstFile = dstDir + newNum + type
            print (dstFile)
            shutil.copy(srcFile, dstFile)
            
    # suburban
    for num in suburbanNum:
        newNum = '1' + str(count)
        count = count + 1
        
        for type in fileType:
            srcFile = srcDir + str(num) + type
            dstFile = dstDir + newNum + type
            print (dstFile)
            shutil.copy(srcFile, dstFile)
    
    # urban
    for num in urbanNum:
        newNum = '2' + str(count)
        count = count + 1
        
        for type in fileType:
            srcFile = srcDir + str(num) + type
            dstFile = dstDir + newNum + type
            print (dstFile)
            shutil.copy(srcFile, dstFile)

tifClassify('/home/jky/jky/tool/dataprocess/VOCdevkit_fujianNew/txt/', '/home/jky/jky/tool/dataprocess/VOCdevkit_fujianNew/txt_classify/')