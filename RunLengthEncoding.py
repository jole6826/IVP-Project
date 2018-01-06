'''
Runleght Encoding
'''

import numpy as np


def runLength(pic):
    picVector = pic.flatten()

    # run length ######################################################################
    value = picVector[0]
    # ! BEACHTEN: count = Anzahl der Wiederholungen
    #   beim Wiederherstellen -> Anzahl des Wertes = count + 1
    count = 0
    encoded = []
    for index in range(1, len(picVector)):
        if value == picVector[index]:
            count += 1
        else:
            # append = anhaengen
            encoded.append([value, count])
            value = picVector[index]
            count = 0
    encoded_vec = np.array(encoded).reshape(len(encoded) * 2)

    return encoded_vec
