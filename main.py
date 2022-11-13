import cv2
import numpy as np
import skimage.exposure
import matplotlib.pyplot as plt
import imutils
import math
from scipy.stats import norm
import seaborn as sns

## Поиск волокн
img = cv2.imread('im2.bmp')

img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
thresh = 140
ret,thresh_img = cv2.threshold(img_grey, thresh, 250, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 9000]
img_contours = np.zeros(img.shape)

cv2.drawContours(img_contours, contours, -1, (0,255,0), -1)

cv2.imwrite('result.png',img_contours) 

img = cv2.imread('result.png')

## Перевод в черно белую фотографию
lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)

A = lab[:,:,1]

thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)

mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)

result = img.copy()
result = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
result[:,:,3] = mask

cv2.imwrite('black.png', mask)



## Поиск линий
def process(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY)
    img_blur = cv2.GaussianBlur(thresh, (5, 5), 2)
    img_canny = cv2.Canny(img_blur, 0, 0)
    return img_canny

def get_contours(img):
    contours = cv2.findContours(process(img), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # mask = np.zeros_like(img)
    # cv2.drawContours(mask, contours, -1, (0,255,0), -1)
    cnts = imutils.grab_contours(contours)

    # cnt = c[0]
    # extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    # print(extTop)

        # dist = math.hypot(extTop[0] - extBot[0], extTop[1] - extBot[1]) 
        # print(extTop, extBot)

    # cnts = imutils.grab_contours(contours)
    # c = max(cnts, key=cv2.contourArea)
    # cv2.imwrite('fin.png', mask)
    # x = []
    # for i in enumerate(contours):
    #     print(i)
    #     # x.append(cv2.contourArea(counter))
    # return x


def count_dist(img):
    height, width, channels = img.shape
    print(height,width)
    middle = width//2
    start = 0
    dists =[]
    all = []
    counting = True
    pixels = 0
    while middle != width:
        while start != height:
            r,g,b = img[start, middle]
            if r != 0 and g != 0 and b !=0:
                if counting:
                    pixels+=1
                else:
                    counting=True
                    pixels = 1
            else:
                if counting:
                    counting = False
                    dists.append(pixels)
            start+=1
        all.append(dists)
        middle+=1
    return all

def remove_arrs(dists):
    dct = {}
    for dist in dists:
        if dct.get(len(dist)):
            dct[len(dist)] = dct[len(dist)] + 1
        else:
           dct[len(dist)] = 1
    dsts = []
    inverse = [(value, key) for key, value in dct.items()]
    key = max(inverse)[1]
    for dist in dists:
        if len(dist) == key:
            dsts.append(dist)

    return dsts

img1 = cv2.imread('black.png')
x = count_dist(img1)

x = remove_arrs(x)
# x = get_contours(img1)
# x = np.linspace(1,50,200)

def normal_dist(x , mean , sd):
    prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

mean = np.mean(x)
sd = np.std(x)
distr = norm(mean, sd)
sns.distplot(distr.rvs(1000))

plt.show() 
