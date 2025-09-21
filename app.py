from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from utils.transcription import transcribe_audio
from utils.summarizer import summarize_text, extract_key_topics
from utils.question_answer import generate_questions, answer_question
from utils.youtube_recommend import get_topic_videos
import warnings
import traceback
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store session data (in production, use proper session management)
sessions = {}

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_cleanup_file(filepath):
    """Safely cleanup uploaded files"""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleaned up file: {filepath}")
    except Exception as e:
        print(f"Warning: Could not cleanup file {filepath}: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    filepath = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Supported: mp3, wav, ogg, m4a, flac, aac'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{session_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Process the audio with error handling
        print(f"Transcribing audio file: {filepath}")
        transcript = None
        try:
            transcript = transcribe_audio(filepath)
        except Exception as e:
            print(f"Transcription error: {e}")
            traceback.print_exc()
            return jsonify({'error': f'Transcription failed: {str(e)}'}), 500
        
        if not transcript or "Transcription failed" in transcript:
            safe_cleanup_file(filepath)
            return jsonify({'error': 'Failed to transcribe audio. Please check the file format and try again.'}), 500
        
        print(f"Transcription successful. Length: {len(transcript)} characters")
        
        # Generate summary with error handling
        print(f"Generating summary...")
        summary = "Summary generation failed"
        try:
            summary = summarize_text(transcript)
        except Exception as e:
            print(f"Summary error: {e}")
            summary = "Unable to generate summary at this time."
        
        # Extract key topics with error handling
        print(f"Extracting key topics...")
        topics = []
        try:
            topics = extract_key_topics(transcript)
        except Exception as e:
            print(f"Topic extraction error: {e}")
            topics = ["General Topic", "Key Concepts"]
        
        # Generate questions with error handling
        print(f"Generating questions...")
        questions = []
        try:
            questions = generate_questions(transcript)
        except Exception as e:
            print(f"Question generation error: {e}")
            questions = [
                "What is the main topic of this lecture?",
                "What are the key concepts discussed?",
                "What examples are provided?",
                "What conclusions are drawn?"
            ]
        
        # Get YouTube recommendations with error handling
        print(f"Getting YouTube recommendations for topics: {topics}")
        recommendations = []
        try:
            for topic in topics[:3]:  # Limit to top 3 topics
                videos = get_topic_videos(topic)
                recommendations.extend(videos)
        except Exception as e:
            print(f"YouTube recommendation error: {e}")
            recommendations = []
        
        # Store session data
        sessions[session_id] = {
            'transcript': transcript,
            'summary': summary,
            'topics': topics,
            'questions': questions,
            'recommendations': recommendations[:6],  # Limit to 6 videos
            'filepath': filepath
        }
        
        return jsonify({
            'session_id': session_id,
            'transcript': transcript,
            'summary': summary,
            'topics': topics,
            'questions': questions,
            'recommendations': recommendations[:6]
        })
        
    except Exception as e:
        print(f"Error processing upload: {str(e)}")
        traceback.print_exc()
        safe_cleanup_file(filepath)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        question = data.get('question')
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        transcript = sessions[session_id]['transcript']
        
        try:
            answer = answer_question(question, transcript)
        except Exception as e:
            print(f"Error answering question: {e}")
            answer = "I'm sorry, I couldn't process your question at this time. Please try rephrasing it or ask a different question."
        
        return jsonify({'answer': answer})
        
    except Exception as e:
        print(f"Error in ask_question: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Failed to answer question: {str(e)}'}), 500

@app.route('/get_more_videos', methods=['POST'])
def get_more_videos():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        topic = data.get('topic')
        
        if not session_id or session_id not in sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        if not topic:
            return jsonify({'error': 'No topic provided'}), 400
        
        try:
            videos = get_topic_videos(topic, max_results=10)
        except Exception as e:
            print(f"Error getting more videos: {e}")
            videos = []
        
        return jsonify({'videos': videos})
        
    except Exception as e:
        print(f"Error in get_more_videos: {str(e)}")
        return jsonify({'error': f'Failed to get videos: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Clean up old files periodically (basic implementation)
@app.route('/cleanup', methods=['POST'])
def cleanup():
    try:
        cleaned_count = 0
        for session_id, session_data in list(sessions.items()):
            if 'filepath' in session_data:
                safe_cleanup_file(session_data['filepath'])
                cleaned_count += 1
        
        sessions.clear()  # Clear all sessions
        return jsonify({'message': f'Cleanup completed. Cleaned {cleaned_count} files.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found.'}), 404

if __name__ == '__main__':
    print("Starting Lecturyzer server...")
    print("Make sure you have run the following commands:")
    print("1. python -m spacy download en_core_web_sm")
    print("2. python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('punkt_tab')\"")
    app.run(debug=True, host='0.0.0.0', port=5000)
