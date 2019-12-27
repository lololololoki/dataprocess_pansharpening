#!/usr/bin/env python
# coding:utf-8

#from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString
import os
import cv2
def genXML(jpg,txt,out):
    lines=open(txt).readlines()
    filename=jpg.split("/")[-1]
    print filename
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'VOC0712'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = '512'

    node_height = SubElement(node_size, 'height')
    node_height.text = '512'

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for line in lines:
        array=line.split()
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'building'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = 'Right'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = array[0]
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = array[1]
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = array[2]
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = array[3]

    xml = tostring(node_root, pretty_print=True)  # ��ʽ����ʾ���û��еĻ���
    xml = xml.replace(" ","")
    dom = parseString(xml)

    output=open(out,"wb")
    output.write(xml)



def batTXT2XML(dir):
    for parent, dirnames, filenames in os.walk(dir + '/boxes'):
        num=len(filenames)
    print "num:%d" % num
    for i in range(num):
        print i
        genXML('%s/JPEGImages/%06d.jpg'%(dir,i),'%s/boxes/%06d.txt'%(dir,i),'%s/Annotations/%06d.xml'%(dir,i))

#batTXT2XML('/data/buildings/VOCdevkit/guizhou')
batTXT2XML('../VOCdevkit/VOC0712')