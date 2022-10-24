import cv2
import pytesseract
from PIL import Image

im_file = "C:/Users/user/Desktop/bigeye/image006.png"

im = Image.open(im_file)
#print(im)
#im.show()
#can crop, rotate with .rotate(90)
#im.save("temp/image006.png")
