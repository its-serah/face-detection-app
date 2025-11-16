import os
import base64
import io
from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize model
model = None

def load_model():
    global model
    if model is None:
        try:
            model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
            model = YOLO(model_path)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_faces():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run detection
        results = model(img)
        
        # Draw bounding boxes
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        face_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline='#00ff00', width=3)
                
                # Draw confidence text
                text = f'{confidence:.2f}'
                draw.text((x1, y1 - 10), text, fill='#00ff00')
                face_count += 1
        
        # Convert to base64
        buffered = io.BytesIO()
        img_draw.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'face_count': face_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect-webcam', methods=['POST'])
def detect_webcam():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run detection
        results = model(img)
        
        # Draw bounding boxes
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        
        face_count = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                draw.rectangle([x1, y1, x2, y2], outline='#00ff00', width=3)
                text = f'{confidence:.2f}'
                draw.text((x1, y1 - 10), text, fill='#00ff00')
                face_count += 1
        
        # Convert to base64
        buffered = io.BytesIO()
        img_draw.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_str}',
            'face_count': face_count
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
