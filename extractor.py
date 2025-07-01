import os
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import PyPDF2
import docx
import pandas as pd
import cv2
import numpy as np

# ✅ Windows path to Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 🧠 OCR config (best for structured typed prescriptions)
OCR_CONFIG = r'--oem 3 --psm 6'  # LSTM with uniform block

# 🔧 Smart preprocessing using OpenCV for clean typed docs
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 🔹 Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # 🔹 Adaptive threshold for better contrast
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 🔹 Save temp processed image
    processed_path = image_path.replace(".", "_processed.")
    cv2.imwrite(processed_path, thresh)

    return processed_path

# 🧾 OCR extraction from image
def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} not found!")

    # Preprocess image for typed OCR
    processed_path = preprocess_image(image_path)

    # OCR with better config
    text = pytesseract.image_to_string(Image.open(processed_path), config=OCR_CONFIG).strip()

    if len(text) < 15:
        return "⚠️ OCR failed. Image may be blurry or too noisy."

    return text

# 📄 PDF Extraction
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
    
    if not text.strip():
        return "⚠️ PDF appears to be scanned or empty. Try using image format instead."

    return text.strip()

# 📃 DOCX
def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

# 📊 CSV
def extract_text_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df.to_string(index=False)

# 📈 Excel
def extract_text_from_excel(excel_path):
    df = pd.read_excel(excel_path)
    return df.to_string(index=False)

# 📜 TXT
def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

# 🧠 File Type Dispatcher
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext in ['.png', '.jpg', '.jpeg']:
            return extract_text_from_image(file_path)
        elif ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif ext == '.docx':
            return extract_text_from_docx(file_path)
        elif ext == '.csv':
            return extract_text_from_csv(file_path)
        elif ext in ['.xls', '.xlsx']:
            return extract_text_from_excel(file_path)
        elif ext == '.txt':
            return extract_text_from_txt(file_path)
        else:
            return "❌ Unsupported file type."
    except Exception as e:
        return f"❌ Error extracting text: {str(e)}"

# ✅ Manual Test
if __name__ == "__main__":
    path = "TestPres3.png"  # Your test file
    output = extract_text_from_file(path)
    print("🧾 Extracted Text:\n", output)
