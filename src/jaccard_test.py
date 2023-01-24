from sklearn.metrics import jaccard_score
import dullrazor
import superpixel
import cProfile
from skimage import io
from os import listdir
from skimage.segmentation import chan_vese
import cv2
import csv

DATASET_PATH = "/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_Data"
GROUNDTRUTH_PATH = "/home/dv/files/2022-09_multimedia/datasets/ISBI2016_ISIC_Part1_Test_GroundTruth"


def get_otsu_thresholded(superpixelized):
    _, thresholded = cv2.threshold(
        superpixelized, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (thresholded == 0) * 255


def get_chan_vese(superpixilized):
    return chan_vese(superpixilized, max_num_iter=60) * 255


def main():
    # Lê o folder com o dataset
    images = listdir(DATASET_PATH)
    # Sorteia a leitura (pode vir fora de ordem)
    images.sort()
    # images = images[0:16]

    with open('jaccard-results.csv', 'w') as csvfile:
        # Inicia a escrita dos resultados em um arquivo .csv
        results_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        # Escreve cabeçalho
        results_writer.writerow(
            ["Image", "Superpixel amount", "Otsu JAC", "Chan-vese JAC", "JAC Product"])

        # Para cada imagem...
        for image in images:
            try:
                print(f'Analyzing {image}')

                # Carrega imagem e sua ground truth equivalente
                pic_from_disk = io.imread(f'{DATASET_PATH}/{image}')
                groundtruth = io.imread(
                    f'{GROUNDTRUTH_PATH}/{image.split(".")[0]}_Segmentation.png')

                # Aplica o DullRazor na imagem (implementado em dullrazor.py)
                hair_removed = dullrazor.dull_razor_on_cv2_img(pic_from_disk)
                cv2.imwrite(f'output/{image}_hair-removed.png', hair_removed)

                for superpixel_qty in range(200, 1600, 50):
                    print(
                        f'Analyzing {image} for {superpixel_qty} superpixels')

                    # Aplica uma superpixalização na imagem com uma quantidade específica.
                    # Divide-se por 255 pois a função exige imagens com valores de cor normalizadas.
                    superpixelized = superpixel.superpixelize_img(
                        hair_removed / 255, superpixel_qty)

                    # Aplica-se Otsu e Chan-Vese
                    otsu = get_otsu_thresholded(superpixelized)
                    chan_v = get_chan_vese(superpixelized)

                    cv2.imwrite(
                        f'output/{image}-{superpixel_qty}-otsu.png', otsu)
                    cv2.imwrite(
                        f'output/{image}-{superpixel_qty}-chan.png', chan_v)

                    # Calcula-se a Score de Jaccard para os resultados de Otsu e Chan-Vese
                    # "micro" inclui falsos positivos no cálculo
                    otsu_score = jaccard_score(
                        groundtruth, otsu, average="micro")
                    chan_vese_score = jaccard_score(
                        groundtruth, chan_v, average="micro")

                    print(
                        f'Writing {image} results for {superpixel_qty} superpixels')

                    # Escreve resultados
                    results_writer.writerow(
                        [image, superpixel_qty, otsu_score, chan_vese_score, otsu_score * chan_vese_score])

            except KeyboardInterrupt:
                exit()
            except Exception:
                continue



if __name__ == "__main__":
    cProfile.run('main()')
