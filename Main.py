import os
import cv2
import lib.encoding_framework as ef
import lib.decoding_framework as df


dumpHuffman = True
dumpRGB = True
pathToFile = os.path.dirname(__file__)
pathToImgs = os.path.join(pathToFile,"imgs")
wallaby = "wallaby.png"
lena = "lena512color.tiff"
moon = 'MoonImage.tif'

imgPath = os.path.join(pathToImgs, lena)
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
loR = 0
loC = 0
width = 512
height = width

# Crop for moon
# loR = 0
# loC = 0
# width = maxWidth
# height = maxHeight
assert (loR <= maxWidth and loC <= maxHeight and (loR+height) <= maxHeight and (loC+width) <= maxWidth)
img = img[loR:loR+height,loC:loC+width,:]

# Encoding using Encoding Framework
outPath = os.path.join(pathToFile,"bin","lena")
#ef.encodeWithHuffman(img, outPath, dumpHuffman=True, dumpRGB=True)


# decoding with Decoding Framework
decodedImg = df.decodeWithHuffman(outPath)


cv2.imshow('original', img)
cv2.waitKey(0)
cv2.imshow('decoded', decodedImg)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.destroyAllWindows()







