import numpy as np
import cv2
import sys
import math

def imageGaurd(i,j):
    if imgOut.shape[0] - 1 < i:
        return 0
    if imgOut.shape[1] - 1 < j:
        return 0
    if 0 > i:
        return 0
    if 0 > j:
        return 0
    return 1

def distanceFromCenter(i,j):
    return math.sqrt((inCenter[0]-i)**2+(inCenter[1]-j)**2)


def showImage(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
    

def undestortFactor(i,j, imageSize):
    radius = imageSize
    parameter = distanceFromCenter(i,j)/radius
    value = 1/math.cos(parameter)#*parameter #radius*angle #2*math.pi/360
    return value


imgInName = 'IMAG0018.JPG'
img = cv2.imread(imgInName, cv2.IMREAD_COLOR)


if img is None:
    exitString = "No file named '" + imgInName + "' was found."
    sys.exit(exitString)

inCenter = [0,0]
xCent = img.shape[0]/2
yCent = img.shape[1]/2
maxSize = img.shape[0]
if img.shape[0] < img.shape[1]:
    maxSize = img.shape[1]


print	"Image input:"
print " name: ", imgInName
print " rows: ", img.shape[0]
print " columns: ", img.shape[1]
print " channels: ", img.shape[2]
print " the center is: (", xCent, ", ", yCent, ")" 

imgOut = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#np.arrange(img.size).reshape((img.shape[0], img.shape[1])) #img.copy()

for i in range(-xCent, xCent):
    for j in range(-yCent, yCent):
        imageSize = maxSize
        s = math.sqrt(undestortFactor(i,j, imageSize))

        if imageGaurd(s*(i)+xCent, s*(j)+yCent):
            imgOut[s*(i)+xCent, s*(j)+yCent] = img[i+xCent, j+yCent]
"""
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
    	s = 0.5
    	if imageGaurd(i*s,j*s)
        imgOut[i*s,j*s] = img[i,j]


        #np.arrange(img.size).reshape((img.))
"""

imgOutName = 'OutputStretch.png'
print	"Image output:"
print " name: ", imgOutName
print " rows: ", imgOut.shape[0]
print " columns: ", imgOut.shape[1]
print " channels: ", imgOut.shape[2]
cv2.imwrite(imgOutName, imgOut)



#showImage(img, "Before")
#showImage(imgOut, "After")