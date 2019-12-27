# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import cv2
import numpy as np

import torchvision

class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)

        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]

        out = []
        for id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=id)

            for ann in self.coco.loadAnns(ann_ids):
                if len(ann['segmentation']) > 0 and len(ann['bbox']) > 0:
                       # and ann['area'] > 10 and ann['area'] < 20000:  # filter area > 10 and < 20000
                    out.append(id)
                    break

        self.ids = out

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        ###jky###
        self.transforms = transforms
        # self.transforms = None
    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        img_id = self.ids[idx]
        file_name = self.coco.loadImgs(img_id)[0]['file_name']

        # guard against no boxes
        if not boxes:
            raise ValueError("Image id {} ({}) doesn't have boxes annotations!".format(self.ids[idx], anno))

        return file_name, img, boxes, idx, anno

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

root = 'VOCdevkit/VOC0712'
imgpath = os.path.join(root, 'JPEGImages')
ann_file = os.path.join(root, 'voc_0712_trainval.json')
remove_images_without_annotations = False

OUTPUT = os.path.join(root, 'vis')

dataset = COCODataset(ann_file, imgpath, remove_images_without_annotations)

for file_name, img, boxes, idx, anno in dataset:

    print(os.path.join(root, file_name))
    img = np.array(img)
    for box in boxes:
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        polypoints = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        polypoints = np.array([polypoints], dtype=np.int32)

        cv2.polylines(img, polypoints, 1, (253, 255, 10), 5)

    if not os.path.exists(OUTPUT):
        os.mkdir(OUTPUT)
    cv2.imwrite(os.path.join(OUTPUT, file_name), img)


