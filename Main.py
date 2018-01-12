import os
import cv2
import lib.encoding_framework as ef
import lib.decoding_framework as df
import numpy as np
from os.path import basename
import glob



dumpHuffman = True
dumpRGB = True
pathToFile = os.path.dirname(__file__)
# pathToImgs = os.path.join(pathToFile,"imgs")
# wallaby = "wallaby.png"
# lena = "lena512color.tiff"
# moon = 'MoonImage.tif'
#
# imgPath = os.path.join(pathToImgs, lena)
# img = cv2.imread(imgPath)
# shape = img.shape
# if len(shape) == 3:
#     maxHeight = shape[0]
#     maxWidth = shape[1]
#     nChannels = shape[2]
# else:
#     maxHeight = shape[0]
#     maxWidth = shape[1]
#     nChannels = 1
#
# # Crop for wallaby
# # loR = 300
# # loC = 700
# # width = 1024
#
# # Crop for lena
# loR = 0
# loC = 0
# width = 256
# height = width
#
# # Crop for moon
# # loR = 0
# # loC = 0
# # width = maxWidth
# # height = maxHeight
# assert (loR <= maxWidth and loC <= maxHeight and (loR+height) <= maxHeight and (loC+width) <= maxWidth)
# img = img[loR:loR+height,loC:loC+width,:]

files = glob.glob('imgs/s_*.jpg')
ix = 0
for file in files:
    filename = basename(files[0])[:-4]
    img = cv2.imread(file)

    # Encoding using Encoding Framework
    outPath = os.path.join(pathToFile,"bin","test" + str(ix))
    ef.encodeWithHuffman(img, outPath, dumpHuffman=True, dumpRGB=True)
    ef.encodeWithRunLength(img, outPath,dumpRunLength=True)

    # decoding with Decoding Framework
    decodedHuffImg = df.decodeWithHuffman(outPath)
    decodedRlImg = df.decodeWithRunLegth(outPath)
    ix += 1



cv2.imshow('original', img)
cv2.waitKey(0)
cv2.imshow('decoded', decodedRlImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyAllWindows()







