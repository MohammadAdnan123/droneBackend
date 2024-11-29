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

# Database Configuration (Render will automatically set the DATABASE_URL env var)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')  # Render sets this
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Enable Flask error logging
logging.basicConfig(level=logging.DEBUG)

# YOLO Model
model = YOLO('train6/weights/best.pt')  # Path to your model (adjust as needed)

# File Model
class File(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    data = db.Column(db.LargeBinary, nullable=False)  # Store file as binary

@app.before_first_request
def create_tables():
    db.create_all()  # Create tables in the database if they don't exist

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
        # Directly use the file from the database without saving it locally
        temp_path = f'./temp/{file.filename}'
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(new_file.data)

        # Perform YOLO prediction
        unique_dir = str(uuid.uuid4())  # Unique identifier for the processed file
        result = model.predict(temp_path, conf=0.6, save=True, save_dir=f'./runs/detect/{unique_dir}')
        
        # Since we are not storing files locally, we can directly return the processed image
        # We'll use the prediction result and save it to the database.

        # Read the processed image and store it back in the database
        processed_path = f'./runs/detect/{unique_dir}/{file.filename}'
        with open(processed_path, 'rb') as processed_file:
            processed_data = processed_file.read()

        processed_file = File(filename=f"processed_{file.filename}", data=processed_data)
        db.session.add(processed_file)
        db.session.commit()

        # Return the URL to access the processed image
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
