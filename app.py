from flask import Flask, render_template, request, redirect, url_for
import cv2
import pytesseract
import easyocr
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'D:\program files\Tesseract-OCR\tesseract.exe'

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], download_enabled=False)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def process_image(image_path):
    try:
        # Load the image and convert to grayscale
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Enhance contrast
        contrast_img = cv2.convertScaleAbs(blurred, alpha=2, beta=30)

        # Apply Adaptive Thresholding
        adaptive_thresh = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological Transformations
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(adaptive_thresh, kernel, iterations=1)
        morphed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Resize the image for better OCR accuracy
        resized_plate = cv2.resize(morphed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Save the processed image for display
        processed_image_name = 'processed_' + os.path.basename(image_path)
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_image_name)
        
        # Check if the processed image is saved successfully
        success = cv2.imwrite(processed_image_path, resized_plate)
        if success:
            print(f"Processed image saved at {processed_image_path}")
        else:
            print("Error saving processed image.")
            return "Error in OCR processing", None

        # OCR with Tesseract (Primary)
        plate_text_tesseract = pytesseract.image_to_string(
            resized_plate, config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()

        # OCR with EasyOCR if needed
        plate_text_easyocr = reader.readtext(resized_plate, detail=0)
        plate_text = plate_text_tesseract if plate_text_tesseract else (plate_text_easyocr[0] if plate_text_easyocr else "No text detected")
        
        # Convert the processed image path to a URL-friendly format
        processed_image_path = processed_image_path.replace("\\", "/")

        return plate_text, processed_image_path
    except Exception as e:
        print("Error processing image:", e)
        return "Error in OCR processing", None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Save the original uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            image_path = file_path.replace("\\", "/")

            # Process the uploaded image
            plate_text, processed_image_path = process_image(file_path)

            return render_template('index.html', plate_text=plate_text, image_path=image_path, processed_image_path=processed_image_path)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
