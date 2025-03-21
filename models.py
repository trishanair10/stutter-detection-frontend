# We don't need database models for this project since we're not storing user data
# This file is created to maintain the structure from the provided blueprint
# In a future extension, we could use this to store user profiles, uploaded audio files, or analysis history

class StutterAnalysis:
    """
    A simple model class to represent stutter analysis results
    """
    def __init__(self, has_stutter, stutter_type, confidence, audio_features):
        self.has_stutter = has_stutter
        self.stutter_type = stutter_type
        self.confidence = confidence
        self.audio_features = audio_features
        
    def to_dict(self):
        return {
            'has_stutter': self.has_stutter,
            'stutter_type': self.stutter_type,
            'confidence': self.confidence,
            'audio_features': self.audio_features
        }
