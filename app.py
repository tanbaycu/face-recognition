from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# Hàm nhận dạng khuôn mặt sử dụng OpenCV
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image, len(faces)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Đọc ảnh từ file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Nhận dạng khuôn mặt
        processed_image, face_count = detect_faces(image)
        
        # Chuyển đổi ảnh đã xử lý thành base64
        _, buffer = cv2.imencode('.jpg', processed_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': f'data:image/jpeg;base64,{img_str}',
            'face_count': face_count
        })

@app.route('/capture', methods=['POST'])
def capture_image():
    # Nhận dữ liệu ảnh từ camera
    image_data = request.json['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Chuyển đổi ảnh sang định dạng OpenCV
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Nhận dạng khuôn mặt
    processed_image, face_count = detect_faces(opencv_image)
    
    # Chuyển đổi ảnh đã xử lý thành base64
    _, buffer = cv2.imencode('.jpg', processed_image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_str}',
        'face_count': face_count
    })

if __name__ == '__main__':
    app.run(debug=True)