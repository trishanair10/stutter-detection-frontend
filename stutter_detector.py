import numpy as np
import librosa
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define stutter types
STUTTER_TYPES = {
    'repetition': 'Repetition of sounds, syllables, or words',
    'prolongation': 'Sound prolongations',
    'blocks': 'Silent blocks or stops',
    'interjections': 'Fillers and interjections',
    'revisions': 'Revisions or modifications of speech',
    'none': 'No significant stutter detected'
}

# Remedies for different stutter types
REMEDIES = {
    'repetition': [
        'Practice slow speech techniques',
        'Try rhythm-based speech exercises',
        'Use light articulatory contacts when speaking',
        'Practice mindfulness during speech',
        'Consider syllable-timed speech practice'
    ],
    'prolongation': [
        'Practice smooth transitions between sounds',
        'Use gentle onset of voicing techniques',
        'Implement airflow management exercises',
        'Try voluntary prolongation exercises',
        'Practice easy vocal onsets'
    ],
    'blocks': [
        'Practice easy onset of speech',
        'Use pull-outs when feeling a block coming',
        'Implement diaphragmatic breathing techniques',
        'Try cancellations when you experience a block',
        'Practice relaxation techniques for speech muscles'
    ],
    'interjections': [
        'Practice pause placement in speech',
        'Work on identifying and reducing filler words',
        'Use phrasing techniques',
        'Practice speaking with purpose and intention',
        'Record yourself to become aware of interjection patterns'
    ],
    'revisions': [
        'Practice pre-planning sentences',
        'Use visualization techniques before speaking',
        'Work on confidence-building exercises',
        'Practice speaking in shorter, complete phrases',
        'Try structured conversation exercises'
    ],
    'none': [
        'Continue practicing good speech habits',
        'Work on general speech clarity',
        'Practice public speaking for confidence',
        'Consider joining a speaking club like Toastmasters',
        'Learn relaxation techniques to maintain fluency'
    ]
}

def process_audio(audio_path):
    """
    Process the audio file to extract features for stutter detection
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        numpy array: Extracted features from the audio
    """
    try:
        logger.debug(f"Processing audio file: {audio_path}")
        
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        # 1. Mel-frequency cepstral coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        
        # 2. Zero-crossing rate (useful for detecting stops/blocks)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        # 3. Spectral centroid (useful for detecting speech characteristics)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(spectral_centroid)
        
        # 4. RMS energy (useful for detecting breaks and emphasis)
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # 5. Spectral bandwidth (helps identify speech characteristics)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = np.mean(bandwidth)
        
        # 6. Spectral contrast (useful for understanding spectral peaks)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        
        # 7. Tempo estimation (speaking rate can be indicative of stutter)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        # Combine features
        features = np.concatenate([
            mfcc_mean, 
            [zcr_mean, zcr_std],
            [centroid_mean],
            [rms_mean, rms_std],
            [bandwidth_mean],
            contrast_mean,
            [tempo]
        ])
        
        logger.debug(f"Extracted {len(features)} features from audio")
        return features
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise Exception(f"Failed to process audio: {str(e)}")

def classify_stutter(features):
    """
    A simplified stutter detection model
    
    In a production environment, this would be replaced with a properly trained ML model.
    Here we use a rule-based approach for demonstration purposes.
    
    Args:
        features (numpy array): Extracted audio features
        
    Returns:
        dict: Classification results with stutter type and confidence
    """
    # This is a simplified rule-based classifier for demonstration
    # In production, you would use a trained machine learning model
    
    # Sample logic for demonstration - in real life this would be ML-based
    # Using the features to make some basic determinations
    
    # MFCC values can help identify speech patterns
    mfcc_mean = features[0:13]
    
    # Zero-crossing rate can help identify blocks and stops
    zcr_mean = features[13]
    zcr_std = features[14]
    
    # Energy and spectral features help identify speech characteristics
    rms_mean = features[16]
    rms_std = features[17]
    
    # For this demo, we'll use a simple heuristic approach
    # Note: This is NOT a real stutter detection model
    
    # Decision tree-like logic for demonstration
    if zcr_std > 0.1 and rms_std > 0.2:
        # High variation in zero-crossing and energy often indicates blocks
        stutter_type = 'blocks'
        confidence = min(0.9, max(0.6, zcr_std * 2))
    elif np.mean(mfcc_mean[1:5]) > 0.5:
        # Certain MFCC patterns might indicate repetition
        stutter_type = 'repetition'
        confidence = min(0.85, max(0.55, np.mean(mfcc_mean[1:5])))
    elif rms_std < 0.05 and zcr_mean > 0.1:
        # Low energy variation with higher zero-crossing might indicate prolongation
        stutter_type = 'prolongation'
        confidence = min(0.8, max(0.5, 1 - rms_std * 10))
    elif rms_mean > 0.3 and zcr_std < 0.05:
        # Higher energy with lower zero-crossing variation might indicate interjections
        stutter_type = 'interjections'
        confidence = min(0.75, max(0.45, rms_mean))
    elif rms_std > 0.1 and np.mean(mfcc_mean[8:12]) > 0.3:
        # Certain patterns of energy and MFCCs might indicate revisions
        stutter_type = 'revisions'
        confidence = min(0.7, max(0.4, rms_std * 2))
    else:
        # Default if no clear pattern is found
        stutter_type = 'none'
        confidence = 0.8
    
    # Calculate stutter presence
    has_stutter = stutter_type != 'none'
    
    # Normalize confidence to 0-100% for user display
    display_confidence = round(confidence * 100)
    
    result = {
        'has_stutter': has_stutter,
        'stutter_type': stutter_type,
        'stutter_description': STUTTER_TYPES[stutter_type],
        'confidence': display_confidence
    }
    
    logger.debug(f"Classification result: {result}")
    return result

def get_remedies(stutter_type):
    """
    Get remedy recommendations based on stutter type
    
    Args:
        stutter_type (str): Type of stutter detected
        
    Returns:
        list: List of remedy recommendations
    """
    if stutter_type in REMEDIES:
        return REMEDIES[stutter_type]
    else:
        return REMEDIES['none']
