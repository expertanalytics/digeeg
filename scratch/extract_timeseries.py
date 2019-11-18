# import the necessary packages
import argparse
import imutils
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import sys

img = cv2.imread("../data/scan1.png")                # reads the image

rot_img = imutils.rotate_bound(img,90)      # rotate the image
cropped = rot_img[:870,100:1300]
cropped2 = cropped.copy()

img_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)    # convert "cropped" image to greyscale (needed by THRESHOLD)

blurred = cv2.GaussianBlur(img_gray, (5,5), 0)          # blur image, otherwise we pick up all the millimetric pattern

# A threshold of 150 seems adecuate with a Gaussian blurring of 5x5. It picks up all data points (except where there is no data), and at the same time it avoids most of the millimetric pattern

thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]     # contours finds whatever is white on a black background, so I'm reversing the image


## Try to find the text in the images automatically

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))   # el kernel para DILATE es un rectangulo 4x4
dilate = cv2.dilate(thresh, kernel, iterations=3)

# cv2.imshow('Input', thresh)
# cv2.moveWindow('Input', 0, 0)
# cv2.imshow('Dilation', dilate)

conts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    # find contours on the dilated image to pick up the big blobs
conts = imutils.grab_contours(conts)

cv2.drawContours(cropped, conts, -1, (0,0,255), 1)       # draw the dilated contours on top of "cropped", which is NOT greyscale (otherwise the contours show up in grey, if I plot them on top of the grayscale image)

cv2.imshow('dilation', cropped)
cv2.moveWindow('dilation', 0, 0)


font = cv2.FONT_HERSHEY_SIMPLEX
j = 0
ROI = []
corners_of_ROI = np.zeros(4)
for i in range(0, len(conts)):
    print('----------------')
    area = cv2.contourArea(conts[i])                            # computes the area inside the contour
    print(i, area)
    x,y,w,h = cv2.boundingRect(conts[i])                        # calculates the bounding minimal rectangle out of a point set

    cv2.rectangle(cropped, (x,y), (x+w,y+h), (255,0,0), 2)      # draws rectangles on top of "cropped" (which is not greyscale) using the specified corners that we found earlier

    # write the area of each contour in different colors depending on their area
    if area >= 10000:
        cv2.putText(cropped, str(area), (x,y-5), font, 0.5, (255,0,0), 1, cv2.LINE_AA)   # print the areas inside of each contour (in blue)
    else:
        cv2.putText(cropped, str(area), (x,y-5), font, 0.5, (0,0,0), 1, cv2.LINE_AA)     # print the areas inside of each contour (in black)

    # mask the rectangles we don't want
    if area < 10000:
        thresh[y:y+h, x:x+w] = 0

    # only select those pieces of the original image where the area of the contour is bigger than 10000 and store in ROI
    if area >= 10000:
        wanted_rectangle = thresh[y:y+h, x:x+w]
        j += 1
        cnrs = [x,y,x+w,y+h]
        corners_of_ROI = np.vstack((corners_of_ROI, cnrs))
        ROI.append(wanted_rectangle)

print('There are %i contours bigger than 10000' % j)
print('There are %i ROIs in this scanner' % j)

cv2.imshow('thresh', thresh)
cv2.moveWindow('thresh', 0, 0)

#plt.clf()                                  # clears the figure
plt.close('all')                            # closes all windows

cmx, cmy = 30, 20
plt.figure(figsize=(cmx/2.54,cmy/2.54))     # set size of figure

## Once we have removed the text, we extract the data
X = np.array([])
Y = np.array([])
for jj in range(1,4):

        for xx in range(thresh.shape[1]):
            ymed = np.array([])
            for yy in range(int(corners_of_ROI[jj,1]),int(corners_of_ROI[jj,3])):

                if thresh[yy,xx] == 255:
                    ymed = np.append(ymed, yy)

            if ymed.size == 0:
#                ymed = -999
                ymed = np.nan
            else:
                ymed = int(np.median(ymed))

#            print(xx,ymed)

            X = np.append(X, xx)
            Y = np.append(Y, ymed)

            plt.plot(X, Y, color="red")
            plt.xlim(0, 1200)
            plt.ylim(870,0)

plt.show()



sys.exit()    # para el programa

plt.close('all')




#-------------------------------------------
# Realmente necesito contornos?

cont = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cont = imutils.grab_contours(cont)

cv2.drawContours(cropped, cont, -1, (0,0,255), 1)       # draw the contours on top of "cropped", which is NOT greyscale (otherwise the contours show up in grey, if I plot them on top of the grayscale image)
cv2.imshow('scanner 1', cropped)
cv2.waitKey(0)




#cv2.imshow('original',blurred)
#cv2.moveWindow('original', 0, 0)

# cv2.destroyAllWindows()   # delete all windows

mask = np.zeros((cropped.shape[0],cropped.shape[1]))
cv2.drawContours(mask, cont, -1, (255,255,255), 1)

cv2.imshow('scanner 2', mask)
cv2.moveWindow('scanner 2', 0, 0)

mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY_INV)[1]
cv2.imshow('scanner 3', mask)



sys.exit()    # para el programa




#-------------------------------------------

x=np.arange(1200)
y=np.arange(870)
X, Y = np.meshgrid(x, y)

fig, axis = plt.subplots(1, 1)
def plot_contour(c):
    c = c.reshape(-1, 2)
    x, y = c.T

    axis.plot(x, y)
plot_contour(cont[0])
#C = axis.contour(X, Y, cont[0])


#C = plt.contour(X, Y, cont)
plt.show()
