import pytesseract
from PIL import Image

# Add tesseract to PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\user\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

ready_file = "C:/Users/user/Desktop/bigeye/temp/testimg.jpg"
img = Image.open(ready_file)
ocr_result = pytesseract.image_to_string(img)
print(ocr_result)