import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pprint

jaccard_results = pd.read_csv("2023-01-22_output/jaccard-results.csv")
jaccard_results.columns = jaccard_results.columns.str.replace(" ", "_")

print(jaccard_results.describe())

# análise das imagens
print("Análise de média das imagens")
jaccard_by_image = jaccard_results.groupby(by="Image")

media_por_imagens = jaccard_by_image.mean(numeric_only=True)
print(media_por_imagens)
media_por_imagens.plot.bar(
    y="JAC_Product", title="Média dos resultados das partições Por Imagem")
# o otsu outperforma o chan-vese na maioria dos casos
# O produto em geral não foi alto, mas ISIC_0000371.jpg e ISIC_0010058.jpg foram bons pra ambos
# independentemente, isso indica que em si a necessidade da metodologia indicada no artigo
#
# Existem imagens muito difíceis... A média não é evidente,
# O Chan Vese consegue melhores resultados, enquanto o Otsu os mantém satisfatórios
print("Análise de desvio padrão das imagens")
plt.show()
desvio_padrao_por_imagens = jaccard_by_image.std(numeric_only=True)
print(desvio_padrao_por_imagens)
desvio_padrao_por_imagens.plot.bar(y="JAC_Product",
                                   title="Desvio padrão dos resultados das partições Por Imagem")
plt.show()
# o chan-vese apresenta resultados mais precisos, embora tenham outliers
# o Otsu, por outro lado, não
# o produto em geral apresenta baixa variabilidade nas imagens
#
# Em queral o desvio padrão mantém à níveis satisfatórios e baixos,
# mas o otsu varia mais

jaccard_by_superpixel = jaccard_results.groupby(
    by="Superpixel_amount", sort=True)
print("Analisando a média dos superpixeis")
media_por_superpixel = jaccard_by_superpixel.mean(numeric_only=True)
print(media_por_superpixel)
media_por_superpixel.plot(title="Média dos Resultados por Superpixel")
plt.show()
# o otsu apresentou resultados mais consistentes com os superpixeis,
# mas 400, 450 e 1200 destacam-se por ter as maiores médias e serem os menores valores.
# quanto ao chan-vese, apresentou bem mais inconsistência com até ~25% de variação com o máximo valor...
# O maior em produto foi o 450, 550.
# PORTANTO, da análise de média destaca-se o 450
#
# Quantidades pequenas não ironicamente tem resultados melhores
# ~450 e 500 novamente destaca-se
print("Analisando o desvio padrão dos superpixeis")
desvio_padrao_por_superpixel = jaccard_by_superpixel.std(numeric_only=True)
print(desvio_padrao_por_superpixel)
desvio_padrao_por_superpixel.plot(
    title="Desvio Padrão dos Resultados Por Superpixel")
plt.show()
# a partir de 1000 existe um plateau de desvio padrão
# mas o valor já é alcançado por volta do 450
#
# Consistências: 750-1100 e 1250 - 1550
