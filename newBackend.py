import logging
from ultralytics import YOLO
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import os
import uuid  # For generating unique IDs
from io import BytesIO

app = Flask(__name__)
app.config['DEBUG'] = True
CORS(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Enable Flask error logging
logging.basicConfig(level=logging.DEBUG)

# YOLO Model
model = YOLO('train6/weights/best.pt')

# File Model
class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)  # Store file as binary

@app.before_first_request
def create_tables():
    db.create_all()

# Route for real-time camera detection
def generate_frames(model):
    import cv2
    cap = cv2.VideoCapture(1)  # Default camera (adjust as needed)
    if not cap.isOpened():
        raise Exception("Could not access the camera")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=0.6, save=False)
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    cap.release()

@app.route('/camera-detection', methods=['GET'])
def camera_detection():
    try:
        return Response(generate_frames(model),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        app.logger.error(f"Error in camera detection: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

# Upload and process image
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save uploaded file to database
        new_file = File(filename=file.filename, data=file.read())
        db.session.add(new_file)
        db.session.commit()

        # YOLO Prediction
        temp_path = f'./temp/{file.filename}'
        os.makedirs('./temp', exist_ok=True)
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(new_file.data)

        unique_dir = os.path.join('./runs/detect', str(uuid.uuid4()))
        os.makedirs(unique_dir, exist_ok=True)
        model.predict(temp_path, conf=0.6, save=True, save_dir=unique_dir)

        # Save processed file back to database
        processed_path = os.path.join(unique_dir, file.filename)
        with open(processed_path, 'rb') as processed_file:
            processed_data = processed_file.read()

        processed_file = File(filename=f"processed_{file.filename}", data=processed_data)
        db.session.add(processed_file)
        db.session.commit()

        return jsonify({'url': f'/processed/{processed_file.id}'})
    except Exception as e:
        app.logger.error(f"Error processing image: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

@app.route('/processed/<file_id>', methods=['GET'])
def serve_processed_image(file_id):
    file = File.query.get(file_id)
    if not file:
        return jsonify({'error': 'File not found'}), 404
    return Response(file.data, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
