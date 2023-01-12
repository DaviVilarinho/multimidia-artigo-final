from skimage.segmentation import slic
from skimage import color
import cv2

def superpixelize_img(image, superpixels_amount):
    segments = slic(image, n_segments = superpixels_amount, start_label=1, compactness=30)
    superpixilized = color.label2rgb(segments, image, kind='avg', bg_label=0)
    return (color.rgb2hsv(superpixilized)[:,:,2] * 255).astype('uint8')


def otsu_image_superpixel(superpixelized):
    _,thresholded = cv2.threshold(superpixelized,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return thresholded
