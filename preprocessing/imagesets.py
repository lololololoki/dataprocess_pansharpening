# -*- coding: utf-8 -*-  
import os
import random
def genDatasets(jpgs,outDir):
    for parent, dirnames, filenames in os.walk(jpgs):
        num = len(filenames)

    trainval_percent=0.7 # 0.5
    train_percent=1 # 0.5

    train=open('%s/train.txt'%outDir,"w")
    test=open('%s/test.txt'%outDir,"w")
    trainval=open('%s/trainval.txt'%outDir,"w")
    val=open('%s/val.txt'%outDir,"w")

    set=range(num)
    trainval_set=random.sample(set,int(num*trainval_percent))
    test_set=[item for item in set if item not in trainval_set]
    
    # jky 2019年1月23日09点37分
    trainval_set=[item for item in set if item in trainval_set]
    
    train_set=random.sample(trainval_set,int(len(trainval_set)*train_percent))
    val_set=[item for item in trainval_set if item not in train_set]
    
    # jky 2019年1月23日09点37分
    train_set=[item for item in set if item in train_set]



    for item in trainval_set:
        trainval.write('%06d\n'%item)
    for item in train_set:
        train.write('%06d\n'%item)
    for item in val_set:
        val.write('%06d\n'%item)
    for item in test_set:
        test.write('%06d\n' % item)

    trainval.close()
    train.close()
    val.close()
    test.close()


genDatasets('../VOCdevkit/VOC0712/JPEGImages','../VOCdevkit/VOC0712/ImageSets/Main')