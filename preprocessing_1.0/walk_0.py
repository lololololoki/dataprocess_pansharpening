# coding:utf-8
import os
def mywalk(path):
    filelist=open(os.path.join(path,'filelist.txt'),'w')
    l1s=os.listdir(path)
    tif_files=[]
    shp_files=[]
    for l1 in l1s:
        if os.path.isdir(os.path.join(path,l1)):
            l2s=os.listdir(os.path.join(path,l1))
            for l2 in l2s:
                if os.path.isdir(os.path.join(path,l1,l2)):
                    l3s=os.listdir(os.path.join(path,l1,l2))
                    for item in l3s:
                        if item.endswith('.tif'):
                            tif_files.append(os.path.join(path,l1,l2,item))
                            print ("tif    %s"%item)
                        elif item.endswith('.shp'):
                            shp_files.append(os.path.join(path,l1,l2,item))
                            print ("shp    %s"%item)

    tif_files.sort()
    shp_files.sort()
    print (tif_files.__len__()==shp_files.__len__())
    for i in range(len(tif_files)):
        print (i)
        print (tif_files[i])
        print (shp_files[i])
        filelist.write('%s %s\n'%(tif_files[i],shp_files[i]))

def mywalk2(path):
    filelist=open(os.path.join(path,'filelist.txt'),'w')

    tif_files=[]
    shp_files=[]
    for root, dirs, files in os.walk(path):
        for item in files:
            if item.endswith('.tif') and '多光谱' not in root:
                tif_files.append(os.path.join(root, item))
                print ("tif    %s"%item)
            elif item.endswith('.shp') and '多光谱' not in root:
                shp_files.append(os.path.join(root, item))
                print ("shp    %s"%item)

    tif_files.sort()
    shp_files.sort()
    print (tif_files.__len__()==shp_files.__len__())
    for i in range(len(tif_files)):
        print (i)
        print (tif_files[i])
        print (shp_files[i])
        filelist.write('%s %s\n'%(tif_files[i],shp_files[i]))
mywalk2('../data')