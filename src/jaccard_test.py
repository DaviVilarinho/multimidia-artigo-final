import numpy as np
from sklearn.metrics import jaccard_score
import dullrazor
import superpixel
from skimage import io
from os import listdir
from skimage.segmentation import chan_vese
import cv2

DATASET_PATH="/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_Data"
GROUNDTRUTH_PATH="/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_GroundTruth"
images = listdir(DATASET_PATH)
images = ["ISIC_0000534.jpg"]


def get_otsu_thresholded(superpixelized):
    _, thresholded = cv2.threshold(
        superpixelized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresholded


def get_chan_vese(superpixilized):
    return chan_vese(superpixilized, max_num_iter=60)

for image in images:
    pic_from_disk = io.imread(f'{DATASET_PATH}/{image}')
    groundtruth = io.imread(f'{GROUNDTRUTH_PATH}/{image.split(".")[0]}_Segmentation.png')
    hair_removed = dullrazor.dull_razor_on_cv2_img(pic_from_disk)
    for superpixel_qtdd in range(400, 700, 50):
        superpixelized = superpixel.superpixelize_img(
            hair_removed / 255, superpixel_qtdd)

        print(
            f'OTSU JAC: \t\t{jaccard_score(groundtruth, get_otsu_thresholded(superpixelized), average="micro")}')
        print(
            f'CHAN VESE JAC: \t\t{jaccard_score(groundtruth, get_chan_vese(superpixelized) * 255, average="micro")}')
