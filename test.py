from concurrent.futures import process
from os import remove

import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image

# functions to process r

# import image
image_file = "C:/Users/user/Desktop/bigeye/image006.png"
img = cv2.imread(image_file)

# setup params
thresh1 = 120
thresh2 = 220
#crop_h =
#crop_w =

# display image
#https://stackoverflow.com/questions/28816046/
#displaying-different-images-with-actual-size-in-matplotlib-subplot
def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()

# invert colour
inverted_image = cv2.bitwise_not(img)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/inverted.jpg", inverted_image)
#display("C:/Users/user/Desktop/bigeye/temp/inverted.png")

# binarise image (= make into greyscale,then b&w, with at a threshold)
def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
grey_image = greyscale(inverted_image)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/grey.jpg", grey_image)
#display("C:/Users/user/Desktop/bigeye/temp/grey.jpg")
thresh, im_bw = cv2.threshold(grey_image,thresh1,thresh2, cv2.THRESH_BINARY)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/bw_image.jpg", im_bw)
#display("C:/Users/user/Desktop/bigeye/temp/bw_image.jpg")

# noise removal
def noise_removal(image):
    import numpy as np
    kernel = np.ones((1,1),np.uint8)
    image = cv2.dilate(image,kernel,iterations=1)
    kernel = np.ones((1,1),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,kernel)
    image = cv2.medianBlur(image, 3)
    return (image)
no_noise = noise_removal(im_bw)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/no_noise.jpg",no_noise)
#display("C:/Users/user/Desktop/bigeye/temp/no_noise.jpg")

# dilation & erosion
def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image,kernel,iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

eroded_image = thin_font(no_noise)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/eroded_image.jpg",eroded_image)
#display("C:/Users/user/Desktop/bigeye/temp/eroded_image.jpg")

def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image,kernel,iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

dilated_image = thick_font(no_noise)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/dilated_image.jpg",dilated_image)
#display("C:/Users/user/Desktop/bigeye/temp/dilated_image.jpg")

# deskewing
new = cv2.imread("C:/Users/user/Desktop/bigeye/temp/no_noise.jpg")

#https://becominghuman.ai/how-to-automatically-deskew-straighten-a-text-image-using-opencv-a0c30aed83df
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

fixed = deskew(new)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/rotated_fixed.jpg",fixed)
#display("C:/Users/user/Desktop/bigeye/temp/rotated_fixed.jpg")

# remove border
def remove_borders(image):
    contours,hierachy = cv2.findContours(image, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1] # take biggest bounding box
    x,y,w,h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


no_borders = remove_borders(no_noise)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/no_borders.jpg",no_borders)
#display("C:/Users/user/Desktop/bigeye/temp/no_borders.jpg")

# add missing border
colour = [255,255,255]
top,bottom,left,right = [150]*4
image_with_border = cv2.copyMakeBorder(no_borders,top,bottom,left,right,cv2.BORDER_CONSTANT,value=colour)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/got_borders.jpg", image_with_border)
#display("C:/Users/user/Desktop/bigeye/temp/got_borders.jpg")

# use the OCR fr
#process_image = thick_font(dilated_image)
process_image = remove_borders(no_noise)
process_image = cv2.copyMakeBorder(process_image,top,bottom,left,right,cv2.BORDER_CONSTANT,value=colour)
cv2.imwrite("C:/Users/user/Desktop/bigeye/temp/processed_image.jpg",process_image)

ready_file = "C:/Users/user/Desktop/bigeye/temp/processed_image.jpg"
img = Image.open(ready_file)
ocr_result = pytesseract.image_to_string(img)
print(ocr_result)

#print(pytesseract.image_to_data(Image.open("C:/Users/user/Desktop/bigeye/temp/processed_image.jpg")))

# post process text extracted
