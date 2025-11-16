# Face Detection Web App

A clean, minimal web application for face detection using YOLOv8. Upload images or use your webcam to detect faces in real-time.

## Features

- Upload images for face detection
- Real-time webcam face detection
- Clean black interface
- Green bounding boxes with confidence scores

## Local Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python app.py
```

4. Open your browser and go to `http://localhost:5000`

## Deployment

### Render (Recommended for Free Tier)

1. Create a new account at [render.com](https://render.com)
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` configuration
5. Click "Create Web Service"
6. Wait for deployment to complete (first build takes ~10-15 minutes)

### Railway

1. Create an account at [railway.app](https://railway.app)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway will automatically detect the `Procfile`
5. Add environment variable: `PORT=5000`
6. Deploy

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select "Gradio" as SDK (or use Docker)
3. Upload all files from this directory
4. The space will automatically deploy

### Vercel/Netlify

Note: These platforms work best for static sites. For this Python app, use Render or Railway instead.

## Free Tier Limitations

- **Render**: 750 hours/month, sleeps after 15 min of inactivity
- **Railway**: 500 hours/month, $5 credit
- **Hugging Face Spaces**: Free for CPU inference

## Model

This app uses the [YOLOv8 Face Detection](https://huggingface.co/arnabdhar/YOLOv8-Face-Detection) model from Hugging Face, fine-tuned on 10k+ face images.

## Tech Stack

- Backend: Flask + Gunicorn
- ML: YOLOv8 (Ultralytics)
- Frontend: Vanilla HTML/CSS/JavaScript

## License

AGPL-3.0 (same as the YOLOv8 model)
