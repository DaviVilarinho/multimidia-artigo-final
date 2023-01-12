import numpy as np
from sklearn.metrics import jaccard_score
import dullrazor
import superpixel
from skimage import io
from os import listdir

DATASET_PATH="/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_Data"
GROUNDTRUTH_PATH="/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_GroundTruth"
images = listdir(DATASET_PATH)
images = ["ISIC_0000534.jpg"]

for image in images:
    pic_from_disk = io.imread(f'{DATASET_PATH}/{image}')
    groundtruth = io.imread(f'{GROUNDTRUTH_PATH}/{image.split(".")[0]}_Segmentation.png')
    hair_removed = dullrazor.dull_razor_on_cv2_img(pic_from_disk)
    for superpixel_qtdd in range(400, 500, 50):
        superpixelized = superpixel.superpixelize_img(hair_removed / 255, superpixel_qtdd)
        otsu_proposed = superpixel.otsu_image_superpixel(superpixelized)

        jscore = jaccard_score(groundtruth, otsu_proposed, average='micro')
        print(jscore)
