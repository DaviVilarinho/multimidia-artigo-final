from sklearn.metrics import jaccard_score
import dullrazor
import superpixel
from skimage import io
from os import listdir
from skimage.segmentation import chan_vese
import cv2
import csv

DATASET_PATH = "/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_Data"
GROUNDTRUTH_PATH = "/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_GroundTruth"

images = listdir(DATASET_PATH)
images = ["ISIC_0000534.jpg", "ISIC_0011333.jpg"]


def get_otsu_thresholded(superpixelized):
    _, thresholded = cv2.threshold(
        superpixelized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresholded


def get_chan_vese(superpixilized):
    return chan_vese(superpixilized, max_num_iter=60)


with open('jaccard-results.csv', 'w') as csvfile:
    results_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

    results_writer.writerow(
        ["Image", "Superpixel amount", "Otsu JAC", "Chan-vese JAC", "JAC Product"])

    for image in images:
        print(f'Analyzing {image}')
        pic_from_disk = io.imread(f'{DATASET_PATH}/{image}')
        groundtruth = io.imread(
            f'{GROUNDTRUTH_PATH}/{image.split(".")[0]}_Segmentation.png')
        hair_removed = dullrazor.dull_razor_on_cv2_img(pic_from_disk)
        for superpixel_qty in range(200, 1600, 50):
            print(f'Analyzing {image} for {superpixel_qty} superpixels')
            superpixelized = superpixel.superpixelize_img(
                hair_removed / 255, superpixel_qty)

            otsu_score = jaccard_score(groundtruth, get_otsu_thresholded(
                superpixelized), average="micro")  # micro because we want fp sp
            chan_vese_score = jaccard_score(groundtruth, get_chan_vese(
                superpixelized) * 255, average="micro")
            print(f'Writing {image} results for {superpixel_qty} superpixels')
            results_writer.writerow(
                [image, superpixel_qty, otsu_score, chan_vese_score, otsu_score * chan_vese_score])
