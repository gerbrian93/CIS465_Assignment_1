import cv2
import numpy as np
from matplotlib import pyplot as plt

 
grid = [[0, 3, 2, 5, 4, 7, 6, 9, 8],
        [3, 0, 1, 2, 3, 4, 5, 6, 7],
        [2, 1, 0, 3, 2, 5, 4, 7, 6],
        [5, 2, 3, 0, 1, 2, 3, 4, 5],
        [4, 3, 2, 1, 0, 3, 2, 5, 4],
        [7, 4, 5, 2, 3, 0, 1, 2, 3],
        [6, 5, 4, 3, 2, 1, 0, 3, 2],
        [9, 6, 7, 4, 5, 2, 3, 0, 1],
        [8, 7, 6, 5, 4, 3, 2, 1, 0]]


grid2 = [[1, 3, 5, 7, 8], 
         [6, 0, 2, 1, 4],
         [8, 1, 4, 2, 3],
         [0, 7, 9, 1, 7],
         [0, 3, 8, 3, 2],
         [0, 4, 2, 6, 1],
         [7, 1, 9, 3, 1],
         [6, 0, 2, 1, 4],
         [8, 1, 4, 2, 3],
         [0, 7, 9, 1, 7],
         [0, 3, 8, 3, 2],
         [0, 4, 2, 6, 1]]


def padgrid(g):
    n = len(g[0])
    x = [0 for _ in range(n + 2)]
    [lst.append(0) for lst in g]
    [lst.insert(0, 0) for lst in g]
    g.insert(0, x)
    g.append(x)
    return g


def changeGrid(g):
    fill = 1
    bilist = []
    while fill < len(g)-1:
        bilist.append([])
        fill += 1
    for i in range(len(g)/2, len(g)-1):
        for j in range(len(g)/2, len(g[i])-1):
            temp = []
            x = g[i][j]
            g1 = g[i][j+1]
            g2 = g[i-1][j+1]
            g3 = g[i-1][j]
            g4 = g[i-1][j-1]
            g5 = g[i][j-1]
            g6 = g[i+1][j-1]
            g7 = g[i+1][j]
            g8 = g[i+1][j+1]
            for z in [g8, g7, g6, g5, g4, g3, g2, g1]:
                temp.append(z)
            bistr = ''
            for element in temp:
                if int(element) - int(x) >= 0:
                    bistr += '1'
                else:
                    bistr += '0'
            bilist[i-1].append(int(bistr, 2))
    return bilist


def changeImage(image):
    fill = 1
    bilist = []
    while fill < len(image)-1:
        bilist.append([])
        fill += 1
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            x = image[i][j]
            g1 = image[i][j+1]
            g2 = image[i-1][j+1]
            g3 = image[i-1][j]
            g4 = image[i-1][j-1]
            g5 = image[i][j-1]
            g6 = image[i+1][j-1]
            g7 = image[i+1][j]
            g8 = image[i+1][j+1]

            bistr = ''
            for k in [g1, g2, g3, g4, g5, g6, g7, g8]:
                if int(k) - int(x) >= 0:
                    bistr += '1'
                else:
                    bistr += '0'
            bilist[i-1].append(int(bistr, 2))
    return np.array(bilist)


hist = [[4, 3, 9],
        [1, 3, 5],
        [9, 1, 7]]


def makeHistogram(g):
    d = {}
    for i in range(0, len(g)):
        for j in range(0, len(g[i])):
            x = g[i][j]
            if x not in d:  # if the word is not in the dictionary
                d[x] = 1
            else:
                d[x] += 1
    d = sorted(k, key=lambda value: value[1], reverse=True)
    for k, v in d:
        print(str(k) + " " + str(v))
        # print(v)

    # return np.array(bilist)


def T1(image):
    fill = 0
    bilist = []
    while fill < len(image):
        bilist.append([])
        fill += 1
    for i in range(0, len(image)):
        for j in range(0, len(image[i])):
            x = image[i][j]

            if x < 70 and 170:
                x = 0
            elif x >= 70 and 170:
                x = 255 - x
            bilist[i].append(x)

    return np.array(bilist)


def makehist(img):
    row, col = img.shape
    y = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            y[img[i, j]] += 1

    x = np.arange(0, 256)
    plt.bar(x, y, color="blue", align="center")
    plt.show()


image = cv2.imread("GerhartWorkPhoto.png", 2)


resize = cv2.resize(image, [500, 700], interpolation=cv2.INTER_AREA)
newimage = T1(resize)
makehist(resize)
makehist(newimage)


#output = changeImage(image)
#output = output.astype(np.uint8)

# cv2.imshow('myimage', final)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
