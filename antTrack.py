from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage import color
from skimage import measure
import scipy.ndimage as ndimage
from random import randint
import math
import KalmanFilter as kf


# List all images
mypath = "antsImages/"
antsImages = [f for f in listdir(mypath) if isfile(join(mypath, f))]
antsImages = antsImages[: -1]
antsImages.sort(key=lambda f: int(filter(str.isdigit, f)))

allImages = [] # Read Images in gray scale
frameStates = [] # state of every object in every frame (rows: frames , cols: objects)
## list of random colors
colors = []
# for i in range(17):
#     colors.append('%06X' % randint(0, 0xFFFFFF))
# from matplotlib import colors as mcolors
#
# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
colors = ['#4B0082', '#FFD700', '#B22222', '#CD5C5C', '#556B2F', '#708090', '#2F4F4F', '#FFEFD5', '#000000', '#FFA07A', '#40E0D0', '#C0C0C0', '#8A2BE2', '#FFB6C1', '#D8BFD8', '#696969', '#008000', '#FAF0E6'
                                                                                                                                                                                                      '#FF6347', '#AFEEEE']
statePredList = []
for imgNum in range (len(antsImages)):
    imgColor = io.imread(mypath + antsImages[imgNum])
    img = color.rgb2gray(imgColor)

    # Compute Histogram of the image
    hist, bins_center = exposure.histogram(img)

    # Thresholding the image
    binaryImg = img < 0.3

    # Morphological dilation
    binaryImg = ndimage.binary_dilation(binaryImg).astype(binaryImg.dtype)

    # Connected Component Labeling
    all_objects  = measure.label(binaryImg, neighbors=8, background=0)

    # lable properties
    objProp = measure.regionprops(all_objects)

    ObjAreas = [] # Valid Objects area
    objIndx = [] # Valid Objects index in all_objects list
    for i in range(len(objProp)):
        objArea = objProp[i].area
        if objArea > 150 and objArea < 1000:
            ObjAreas.append(objArea)
            objIndx.append(i)

    objectsState = [] # state of every object in one frame
    for i in range(len(objIndx)):
        objCentroid = objProp[objIndx[i]].centroid
        objectsState.append(objCentroid)

    frameStates.append(objectsState)

    ######################################
    #Association
    ######################################

    ### Perorm objects associations bet. current observations and previous obs.
    prevObjectsState = []
    currObjectsState = []
    newObjectsState = [None]*(17) # After association
    if imgNum > 0:
        prevObjectsState = frameStates[imgNum-1]
        currObjectsState = frameStates[imgNum]
        for i in range(len(currObjectsState)):
            minDist = 10000
            index = 0
            x1 = currObjectsState[i][0]
            y1 = currObjectsState[i][1]
            for j in range(len(prevObjectsState)):
                x2 = prevObjectsState[j][0]
                y2 = prevObjectsState[j][1]
                dist = math.sqrt(((x1 - x2)**2) +  ((y1 - y2)**2))

                if dist < minDist:
                    index = j
                    minDist = dist

            newObjectsState[index] = currObjectsState[i]

        frameStates[imgNum] = newObjectsState
        print imgNum
    ######################################

    ######################################
    # Kalman Filter
    ######################################

    if imgNum > 0:
        r = frameStates[imgNum][13][0]
        c = frameStates[imgNum][13][1]
        meas = np.matrix(frameStates[imgNum][10]).T

        R = 0.01 ** 2
        statePred, Cov = kf.kalman_xy(statePredList[imgNum-1], Cov, meas, R)
        statePredList.append((statePred[:2]).tolist())
        print statePredList[imgNum][0]
        print meas

    else:
        statePred = np.matrix(frameStates[imgNum][10]).T
        Cov = np.matrix(np.eye(2)) * 1000  # initial uncertainty
        statePredList.append((statePred[:2]).tolist())
    ##########################################

    # plt.imshow(all_objects, cmap='spectral')

    my_dpi = 96
    plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)
    plt.imshow(imgColor, cmap='spectral')

    for frameNum in range(len(frameStates)):
        for objNum in range(len(objectsState)):
            row = frameStates[frameNum][objNum][0]
            col = frameStates[frameNum][objNum][1]
            # colorHash = str(colors[objNum])
            plt.scatter(col, row,  c=colors[objNum], s=10)

    for frameNum in range(len(statePredList)):
        if frameNum != 0:
            row = statePredList[frameNum][0][0]
            col = statePredList[frameNum][0][1]
            # plt.scatter(col, row, c=r'', s=30)

    # figName =  "associationPredTrack/" + str(imgNum) + ".png"
    # plt.savefig(figName, dpi=my_dpi)
    # plt.close()
    if imgNum%10 == 0:
        plt.show()
