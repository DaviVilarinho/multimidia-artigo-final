# -*- coding: utf-8 -*-

import cv2
"""
@author: Javier Velasquez P.
Algoritmo DullRazor.
Algumas modificações foram feitas, descritas no artigo
"""
def dull_razor_on_cv2_img(img):
    #Escala de cinza
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    #Black hat filter
    kernel = cv2.getStructuringElement(1,(20,20)) 
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    #Filtro Gaussiano
    bhg= cv2.GaussianBlur(blackhat,(9,9),cv2.BORDER_DEFAULT)
    #Thresholding Binário (transforma em máscara)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    #Troca os pixeis na imagem original em relação à máscara.
    dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)

    return dst
