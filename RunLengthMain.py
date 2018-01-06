
import cv2
import numpy as np
import pickle

from RunLengthEncoding import runLength


pic = cv2.imread('Test.jpg')
pickle.dump(pic, open('test.p','wb'), protocol=pickle.HIGHEST_PROTOCOL)
b = pic[:,:,0]
g = pic[:,:,1]
r = pic[:,:,2]

encoded_vec_B = runLength(b)
encoded_vec_G = runLength(g)
encoded_vec_R = runLength(r)

pickle.dump([encoded_vec_B,encoded_vec_G,encoded_vec_R], open('runLengthBGR.p','wb'), protocol=pickle.HIGHEST_PROTOCOL)