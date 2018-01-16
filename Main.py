import os
import cv2
import lib.encoding_framework as ef
import lib.decoding_framework as df
import lib.helpers as help
import numpy as np
from os.path import basename
import glob
import time

dumpHuffman = True
dumpRGB = True
showImgs = True
pathToFile = os.path.dirname(__file__)

files = glob.glob('imgs/s_*.jpg')
ix = 0


for file in files:
    filename = basename(file)[:-4]

    img = cv2.imread(file)

    # Encoding using Encoding Framework
    outPath = os.path.join(pathToFile,"bin",filename + str(ix))
    t = time.time()
    ef.encodeWithHuffman(img, outPath, dumpHuffman=True, dumpRGB=True)
    elapsed = time.time() - t
    print("HM:" + str(elapsed))
    t = time.time()
    ef.encodeWithRunLength(img, outPath,dumpRunLength=True)
    elapsed = time.time() - t
    print("RL:" + str(elapsed))
    # decoding with Decoding Framework
    decodedHMImg = df.decodeWithHuffman(outPath)
    decodedRLImg = df.decodeWithRunLegth(outPath)

    diffHM = np.abs(img.astype(np.float)-decodedHMImg.astype(np.float))
    diffRL = np.abs(img.astype(np.float)-decodedRLImg.astype(np.float))

    imgEntropy = help.entropy(img, showPlot=False)
    print("Entropy of image: " + filename + " is: " + str(imgEntropy))

    # Display original and decoded images as well as error images
    if showImgs:
        cv2.imshow('original', img)
        cv2.waitKey(0)
        cv2.imshow('Run length decoded', decodedRLImg)
        cv2.waitKey(0)
        cv2.imshow('Huffman decoded', decodedHMImg)
        cv2.waitKey(0)
        cv2.imshow('RL decoded error', diffRL.astype(np.uint8))
        cv2.waitKey(0)
        cv2.imshow('Huffman decoded error', diffHM.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ix += 1










