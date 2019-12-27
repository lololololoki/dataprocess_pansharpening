cd /home/jky/jky/tool/dataprocess_test
rm -r VOCdevkit
mkdir -p VOCdevkit/txt
mkdir -p VOCdevkit/VOC0712/Annotations
mkdir -p VOCdevkit/VOC0712/boxes
mkdir -p VOCdevkit/VOC0712/ImageSets/Main
mkdir -p VOCdevkit/VOC0712/JPEGImages
mkdir -p VOCdevkit/VOC0712/points
cd preprocessing
bash run.sh