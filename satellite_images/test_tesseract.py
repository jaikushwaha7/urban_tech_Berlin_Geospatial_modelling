import os

# Add Tesseract to PATH
tesseract_path = r'C:\Program Files\Tesseract-OCR'
if tesseract_path not in os.environ['PATH']:
    os.environ['PATH'] = tesseract_path + os.pathsep + os.environ['PATH']

import pytesseract
pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print("Tesseract path:", pytesseract.pytesseract.pytesseract_cmd)
print("Tesseract version:", pytesseract.get_tesseract_version())
