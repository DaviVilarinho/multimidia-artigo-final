import pandas as pd

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


jaccard_results = pd.read_csv("jaccard-results.csv")
jaccard_results.columns = jaccard_results.columns.str.replace(" ", "_")

basic_tendencies = jaccard_results.describe()
print(f"{bcolors.OKBLUE}\nTendencias Basicas=\n{basic_tendencies}{bcolors.ENDC}")

by_image = jaccard_results.groupby(by="Image")
mean = by_image.mean(numeric_only=True)
print(f"{bcolors.OKCYAN}\nMedia por Imagem=\n{mean}{bcolors.ENDC}")

stddev = by_image.std(numeric_only=True)
print(f"{bcolors.OKGREEN}\nDesvio por Imagem=\n{stddev}{bcolors.ENDC}")


by_superpixel = jaccard_results.groupby(by="Superpixel_amount", sort=True)
mean2 = by_superpixel.mean(numeric_only=True)
print(f"{bcolors.BOLD}\nMedia por Superpixel=\n{mean2}{bcolors.ENDC}")

stddev2 = by_superpixel.std(numeric_only=True)
print(f"{bcolors.HEADER}\nDesvio por Superpixel=\n{stddev2}{bcolors.ENDC}")
