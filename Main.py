import os
import cv2
import lib.encoding_framework as ef
import lib.decoding_framework as df
import numpy as np


dumpHuffman = True
dumpRGB = True
pathToFile = os.path.dirname(__file__)
pathToImgs = os.path.join(pathToFile,"imgs")
wallaby = "wallaby.png"
lena = "lena512color.tiff"
moon = 'MoonImage.tif'

imgPath = os.path.join(pathToImgs, moon)
img = cv2.imread(imgPath)
shape = img.shape
if len(shape) == 3:
    maxHeight = shape[0]
    maxWidth = shape[1]
    nChannels = shape[2]
else:
    maxHeight = shape[0]
    maxWidth = shape[1]
    nChannels = 1

# Crop for wallaby
# loR = 300
# loC = 700
# width = 1024

# Crop for lena
# loR = 0
# loC = 0
# width = 512
# height = width

# Crop for moon
loR = 0
loC = 0
width = maxWidth
height = maxHeight
assert (loR <= maxWidth and loC <= maxHeight and (loR+height) <= maxHeight and (loC+width) <= maxWidth)
img = img[loR:loR+height,loC:loC+width,:]

test=np.zeros([10,10])

# Encoding using Encoding Framework
outPath = os.path.join(pathToFile,"bin","moon")
ef.encodeWithHuffman(img, outPath, dumpHuffman=True, dumpRGB=True)
ef.encodeWithRunLength(img,outPath,dumpRunLength=True)

# decoding with Decoding Framework
decodedHuffImg = df.decodeWithHuffman(outPath)
decodedRlImg = df.decodeWithRunLegth(outPath)



cv2.imshow('original', img)
cv2.waitKey(0)
cv2.imshow('decoded', decodedHuffImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyAllWindows()







