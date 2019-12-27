#!/usr/bin/env bash
cd /home/jky/jky/tool/dataprocess_pansharpening
rm -r VOCdevkit
mkdir -p VOCdevkit/txt
mkdir -p VOCdevkit/VOC0712/Annotations
mkdir -p VOCdevkit/VOC0712/boxes
mkdir -p VOCdevkit/VOC0712/ImageSets/Main
mkdir -p VOCdevkit/VOC0712/JPEGImages
mkdir -p VOCdevkit/VOC0712/points
mkdir -p VOCdevkit/VOC0712/mul
mkdir -p VOCdevkit/VOC0712/pan
cd preprocessing
bash run.sh
matlab -nojvm -nodesktop -nosplash -nodisplay -r VOC2coco
exit