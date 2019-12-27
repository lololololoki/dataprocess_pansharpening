source activate caffe2
cd /data/dataprocess
rm -r VOCdevkit
mkdir VOCdevkit
cd VOCdevkit
mkdir txt
mkdir VOC0712
cd VOC0712
mkdir Annotations
mkdir boxes
mkdir ImageSets
mkdir JPEGImages
mkdir points
cd ImageSets
mkdir Main
cd /data/dataprocess/preprocessing
bash run.sh
#matlab -nodesktop -r VOC2coco
matlab -nojvm -nodesktop -nosplash -nodisplay -r VOC2coco  
exit