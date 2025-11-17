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
            print("Starting model download...")
            # Use lighter YOLOv8n model instead of full model
            model = YOLO('yolov8n-face.pt')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            model = None
    return model

def get_model():
    global model_loaded
    load_model()
    if model is None:
        raise RuntimeError("Model failed to load")
    model_loaded = True
    return model

model_loading = False
model_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded}), 200

@app.route('/detect', methods=['POST'])
def detect_faces():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load model on first request
        detection_model = get_model()
        
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize if too large
        max_size = 1280
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Run detection
        results = detection_model(img, conf=0.5)
        
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
        print(f'Error in detect_faces: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

@app.route('/detect-webcam', methods=['POST'])
def detect_webcam():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Load model on first request
        detection_model = get_model()
        
        # Decode base64 image
        try:
            img_data = data['image'].split(',')[1]
        except IndexError:
            img_data = data['image']
        
        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Resize if too large
        max_size = 1280
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Run detection
        results = detection_model(img, conf=0.5)
        
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
        print(f'Error in detect_webcam: {str(e)}')
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
