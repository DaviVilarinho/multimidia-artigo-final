from skimage.segmentation import slic
from skimage import color

def superpixelize_img(image, superpixels_amount):
    segments = slic(image, n_segments = superpixels_amount, start_label=1, compactness=30)
    superpixilized = color.label2rgb(segments, image, kind='avg', bg_label=0)
    return (color.rgb2hsv(superpixilized)[:,:,2] * 255).astype('uint8')


