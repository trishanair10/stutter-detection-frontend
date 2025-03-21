import os
import logging
import librosa
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
import uuid
from stutter_detector import process_audio, classify_stutter, get_remedies

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev_secret_key")

# Configure upload folder
UPLOAD_FOLDER = 'audio_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp3', 'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'audio_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['audio_file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Process the audio file
            features = process_audio(filepath)
            
            # Classify stutter type and confidence
            stutter_results = classify_stutter(features)
            
            # Get remedies based on stutter type
            remedies = get_remedies(stutter_results['stutter_type'])
            
            # Store results in session for display
            session['analysis_results'] = {
                'has_stutter': stutter_results['has_stutter'],
                'stutter_type': stutter_results['stutter_type'],
                'confidence': stutter_results['confidence'],
                'remedies': remedies,
                'filename': filename,
                'features': features.tolist() if isinstance(features, np.ndarray) else features
            }
            
            # Remove the file after processing
            os.remove(filepath)
            
            return redirect(url_for('results'))
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            flash(f"Error processing audio: {str(e)}")
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload .mp3 or .wav files only.')
        return redirect(url_for('index'))

@app.route('/results')
def results():
    if 'analysis_results' not in session:
        flash('No analysis results available. Please upload an audio file first.')
        return redirect(url_for('index'))
    
    results = session['analysis_results']
    return render_template('results.html', results=results)

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    flash('Server error. Please try again or contact support if the issue persists.')
    return redirect(url_for('index'))

# Clear session when closing the browser
@app.before_request
def make_session_permanent():
    session.permanent = False
