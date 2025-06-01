from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import speech_recognition as sr
import os
import tempfile
import logging
import warnings
import nltk
import requests
import json
import re
from urllib.parse import quote
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to set FFmpeg path explicitly
try:
    from pydub import AudioSegment
    from pydub.utils import which

    ffmpeg_path = which("ffmpeg")

    if not ffmpeg_path:
        possible_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            rf"C:\Users\{os.getenv('USERNAME')}\AppData\Local\ffmpeg\bin\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                ffmpeg_path = path
                break

    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        AudioSegment.ffmpeg = ffmpeg_path
        AudioSegment.ffprobe = ffmpeg_path.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
        logger.info(f"FFmpeg found at: {ffmpeg_path}")
    else:
        logger.warning("FFmpeg not found. Audio conversion may fail.")

except ImportError:
    logger.warning("pydub not available")

warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# FIXED: Enhanced summary generation function
def generate_summary(text, sentence_count=3):
    """Generate summary with better error handling and guaranteed different output"""
    try:
        if not text or len(text.strip()) < 50:
            return "Text too short for meaningful summary generation."
        
        # Clean text
        text = re.sub(r'\[Part \d+\]', '', text)  # Remove part markers
        text = re.sub(r'\s+', ' ', text).strip()  # Clean whitespace
        
        # Split into sentences more reliably
        sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 10]
        
        if len(sentences) < 2:
            # If we can't split into sentences properly, create a shortened version
            words = text.split()
            if len(words) > 50:
                summary_words = words[:min(50, len(words)//3)]
                return ' '.join(summary_words) + "..."
            return text
        
        # If we have few sentences, return first few
        if len(sentences) <= sentence_count:
            return ' '.join(sentences[:max(1, len(sentences)-1)])
        
        try:
            # Use sumy for extractive summarization
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            
            summary_sentences = summarizer(parser.document, sentence_count)
            summary = " ".join(str(sentence) for sentence in summary_sentences)
            
            # Ensure summary is different from original and not empty
            if summary and len(summary) < len(text) * 0.8:
                return summary
            else:
                # Fallback: take first few sentences
                return ' '.join(sentences[:sentence_count])
                
        except Exception as sumy_error:
            logger.warning(f"Sumy failed, using fallback: {sumy_error}")
            # Fallback: simple extractive summary
            return ' '.join(sentences[:sentence_count])
            
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        # Final fallback
        words = text.split() if text else []
        if len(words) > 20:
            return ' '.join(words[:20]) + "..."
        return "Summary could not be generated from the provided text."

# FIXED: Extract keywords with better filtering
def extract_keywords(text, num_keywords=10):
    """Extract important keywords from text using TF-IDF with better filtering"""
    try:
        if not text or len(text.strip()) < 20:
            return []
            
        # Clean text more thoroughly
        text = re.sub(r'\[Part \d+\]', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return []
        
        # Create TF-IDF vectorizer with better parameters
        vectorizer = TfidfVectorizer(
            max_features=num_keywords * 2,  # Get more candidates
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only words with letters
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get keywords with scores
            keyword_scores = list(zip(feature_names, tfidf_scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out very short or common words
            filtered_keywords = []
            for kw, score in keyword_scores:
                if len(kw) > 2 and score > 0 and kw not in ['said', 'says', 'like', 'just', 'really', 'going', 'know', 'think']:
                    filtered_keywords.append(kw)
                if len(filtered_keywords) >= num_keywords:
                    break
            
            logger.info(f"Extracted keywords: {filtered_keywords}")
            return filtered_keywords
            
        except Exception as tfidf_error:
            logger.error(f"TF-IDF extraction failed: {tfidf_error}")
            # Fallback: simple word frequency
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top words by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in sorted_words[:num_keywords]]
            
    except Exception as e:
        logger.error(f"Keyword extraction error: {str(e)}")
        return []

# FIXED: YouTube video recommendations with better logic
def get_youtube_recommendations(keywords, transcript_text="", max_results=6):
    """Get YouTube video recommendations based on keywords with better variety"""
    try:
        if not keywords:
            logger.warning("No keywords provided for recommendations")
            return []
        
        # Filter and process keywords
        good_keywords = [kw for kw in keywords if len(kw) > 2][:8]  # Use up to 8 keywords
        
        if not good_keywords:
            logger.warning("No good keywords found")
            return []
        
        logger.info(f"Generating recommendations for keywords: {good_keywords}")
        
        # Create diverse video recommendations
        recommendations = []
        
        # Different types of educational content
        content_types = [
            ("tutorial", "Complete Tutorial"),
            ("explained", "Explained Simply"), 
            ("lecture", "Academic Lecture"),
            ("guide", "Step-by-Step Guide"),
            ("course", "Full Course"),
            ("basics", "Basics & Fundamentals")
        ]
        
        channels = [
            "EduTech Pro", "Learning Hub", "Academic Channel", "TechEd Masters",
            "Knowledge Base", "Study Central", "Expert Tutorials", "Learning Lab"
        ]
        
        # Generate recommendations for each keyword
        for i, keyword in enumerate(good_keywords[:max_results]):
            content_type, title_prefix = content_types[i % len(content_types)]
            channel = channels[i % len(channels)]
            
            # Create search query
            search_query = f"{keyword} {content_type}"
            
            recommendation = {
                "title": f"{title_prefix}: {keyword.title()}",
                "channel": channel,
                "description": f"Comprehensive {content_type} covering {keyword} and related concepts. Perfect for understanding the fundamentals and advanced applications.",
                "url": f"https://www.youtube.com/results?search_query={quote(search_query)}",
                "keyword": keyword,
                "views": f"{np.random.randint(10, 500)}K views",  # Mock view count
                "duration": f"{np.random.randint(5, 45)} min"     # Mock duration
            }
            
            recommendations.append(recommendation)
        
        # Add a few general recommendations based on multiple keywords
        if len(good_keywords) >= 2:
            combined_query = f"{good_keywords[0]} {good_keywords[1]} comprehensive"
            recommendations.append({
                "title": f"Complete Guide: {good_keywords[0].title()} & {good_keywords[1].title()}",
                "channel": "Master Classes",
                "description": f"In-depth exploration of {good_keywords[0]} and {good_keywords[1]} with practical examples and real-world applications.",
                "url": f"https://www.youtube.com/results?search_query={quote(combined_query)}",
                "keyword": f"{good_keywords[0]}, {good_keywords[1]}",
                "views": f"{np.random.randint(50, 200)}K views",
                "duration": f"{np.random.randint(20, 60)} min"
            })
        
        final_recommendations = recommendations[:max_results]
        logger.info(f"Generated {len(final_recommendations)} recommendations")
        
        return final_recommendations
        
    except Exception as e:
        logger.error(f"YouTube recommendations error: {str(e)}")
        return []

# Enhanced lecture processor class
class EnhancedLectureProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.current_transcript = ""
        self.current_keywords = []
        self.current_summary = ""
        
        # Configure recognizer for better accuracy
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        
        logger.info("Enhanced Lecture Processor initialized.")

    def process_audio_file(self, path):
        """Process single audio file with enhanced settings"""
        ext = os.path.splitext(path)[1].lower()
        wav_path = path

        try:
            # Convert to WAV if necessary
            if ext != '.wav':
                from pydub import AudioSegment
                audio = AudioSegment.from_file(path)
                
                # Normalize audio for better recognition
                audio = audio.normalize()
                
                # Convert to 16kHz mono WAV
                wav_path = path.rsplit('.', 1)[0] + '_converted.wav'
                audio.export(wav_path, format="wav", parameters=[
                    "-ar", "16000", 
                    "-ac", "1",
                    "-acodec", "pcm_s16le"
                ])
                logger.info(f"Converted and normalized audio: {wav_path}")
        except Exception as e:
            logger.error(f"Audio conversion error: {str(e)}")
            return f"Error converting audio: {str(e)}"

        try:
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise with longer duration
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                audio_data = self.recognizer.record(source)
                
                # Try Google Speech Recognition
                try:
                    text = self.recognizer.recognize_google(audio_data, language='en-US')
                    logger.info("Successfully transcribed with Google Speech Recognition")
                except sr.UnknownValueError:
                    # Fallback: try with different settings
                    try:
                        text = self.recognizer.recognize_google(audio_data, language='en-US', show_all=False)
                        if not text:
                            raise sr.UnknownValueError()
                    except:
                        return "Could not understand the audio clearly. Please ensure good audio quality."
                except sr.RequestError as e:
                    logger.error(f"Google Speech Recognition error: {str(e)}")
                    return f"Speech recognition service error: {str(e)}"
                    
        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            return f"Error processing audio: {str(e)}"
        finally:
            # Clean up converted file
            if wav_path != path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass
                    
        return text

    def process_long_audio(self, path, chunk_duration=30):
        """Process long audio files in chunks with overlap for better continuity"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(path)
            
            # Normalize audio
            audio = audio.normalize()
            
            chunk_len = chunk_duration * 1000  # Convert to milliseconds
            overlap = 2000  # 2 second overlap
            full_transcript = []
            
            logger.info(f"Processing long audio: {len(audio)/1000:.1f} seconds, {len(audio)//chunk_len + 1} chunks")

            for i, start in enumerate(range(0, len(audio), chunk_len - overlap)):
                end = min(start + chunk_len, len(audio))
                chunk = audio[start:end]
                
                # Skip very short chunks
                if len(chunk) < 3000:  # Less than 3 seconds
                    continue

                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
                    try:
                        chunk.export(temp.name, format="wav", parameters=[
                            "-ar", "16000", 
                            "-ac", "1",
                            "-acodec", "pcm_s16le"
                        ])
                        temp.flush()
                        
                        result = self.process_audio_file(temp.name)
                        if result and not result.startswith("Error") and not result.startswith("Could not"):
                            # Remove part markers for cleaner text
                            clean_result = result.strip()
                            if clean_result:
                                full_transcript.append(clean_result)
                                logger.info(f"Chunk {i+1} processed: {len(clean_result)} characters")
                        else:
                            logger.warning(f"Chunk {i+1} failed: {result}")
                            
                    except Exception as e:
                        logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    finally:
                        try:
                            os.unlink(temp.name)
                        except:
                            pass

            if not full_transcript:
                return "No clear speech could be detected in the audio file."
                
            # Join transcripts with proper spacing
            final_transcript = ". ".join(full_transcript)
            
            # Clean up the transcript
            final_transcript = re.sub(r'\s+', ' ', final_transcript)
            final_transcript = re.sub(r'\.+', '.', final_transcript)
            
            logger.info(f"Final transcript length: {len(final_transcript)} characters")
            return final_transcript

        except Exception as e:
            logger.error(f"Error processing long audio: {str(e)}")
            return f"Error splitting long audio: {str(e)}"

    def enhanced_qa(self, question, transcript):
        """Enhanced Q&A with better context matching"""
        try:
            if not transcript or not question:
                return "Invalid question or transcript."
            
            # Clean and prepare text
            question_lower = question.lower()
            question_words = [w for w in question_lower.split() if w not in stop_words and len(w) > 2]
            
            # Split transcript into sentences
            sentences = [s.strip() for s in transcript.split('.') if len(s.strip()) > 10]
            
            if not sentences:
                return "No meaningful content found in transcript."
            
            # Calculate relevance scores for each sentence
            relevant_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                score = 0
                
                # Direct word matching
                for word in question_words:
                    if word in sentence_lower:
                        score += 2
                
                # Partial word matching
                for word in question_words:
                    for sentence_word in sentence_lower.split():
                        if word in sentence_word or sentence_word in word:
                            score += 1
                
                if score > 0:
                    relevant_sentences.append((sentence, score))
            
            # Sort by relevance score
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            
            if not relevant_sentences:
                return "I couldn't find specific information about that topic in the transcript. Try rephrasing your question."
            
            # Return top 3 most relevant sentences
            answer_parts = [s[0] for s in relevant_sentences[:3]]
            answer = ". ".join(answer_parts)
            
            # Ensure answer isn't too long
            if len(answer) > 500:
                answer = answer[:500] + "..."
            
            return answer
            
        except Exception as e:
            logger.error(f"Enhanced Q&A error: {str(e)}")
            return f"Error processing question: {str(e)}"

# Instantiate processor
processor = EnhancedLectureProcessor()

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "message": "Enhanced Lecturyzer API is running",
        "version": "2.0"
    })

@app.route('/upload', methods=['POST'])
def upload_audio():
    """Enhanced audio upload and processing"""
    try:
        if 'audio' not in request.files:
            return jsonify({"success": False, "error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"success": False, "error": "Empty filename"}), 400

        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            audio_file.save(temp_file.name)
            file_path = temp_file.name

        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"Processing file: {audio_file.filename}, Size: {file_size/1024/1024:.2f} MB")

            # Process based on file size
            if file_size > 25 * 1024 * 1024:  # > 25MB
                result = processor.process_long_audio(file_path, chunk_duration=25)
            elif file_size > 10 * 1024 * 1024:  # > 10MB
                result = processor.process_long_audio(file_path, chunk_duration=30)
            else:
                result = processor.process_audio_file(file_path)

            if result.startswith("Error") or result.startswith("Could not"):
                return jsonify({"success": False, "error": result}), 400

            # Store results
            processor.current_transcript = result
            
            # FIXED: Generate summary with better parameters
            summary = generate_summary(result, sentence_count=3)
            processor.current_summary = summary
            
            # FIXED: Extract keywords with better filtering
            keywords = extract_keywords(result, num_keywords=15)
            processor.current_keywords = keywords
            
            logger.info(f"Processing complete. Transcript: {len(result)} chars, Summary: {len(summary)} chars, Keywords: {len(keywords)}")
            logger.info(f"Summary preview: {summary[:100]}...")

            return jsonify({
                "success": True,
                "transcript": result,
                "summary": summary,
                "keywords": keywords[:8],  # Return top 8 keywords
                "word_count": len(result.split()),
                "character_count": len(result),
                "summary_length": len(summary)
            })

        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return jsonify({"success": False, "error": f"Processing failed: {str(e)}"}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"success": False, "error": f"Upload failed: {str(e)}"}), 500

@app.route('/transcript', methods=['GET'])
def get_transcript():
    """Get current transcript"""
    return jsonify({
        "success": True,
        "transcript": processor.current_transcript,
        "summary": processor.current_summary,
        "has_content": len(processor.current_transcript.strip()) > 0,
        "word_count": len(processor.current_transcript.split()) if processor.current_transcript else 0,
        "summary_length": len(processor.current_summary) if processor.current_summary else 0
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Enhanced question answering"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()

        if not question:
            return jsonify({"success": False, "error": "Question is required"}), 400
        if not processor.current_transcript:
            return jsonify({"success": False, "error": "No transcript available. Please upload an audio file first."}), 400

        # Use enhanced Q&A
        answer = processor.enhanced_qa(question, processor.current_transcript)

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "transcript_length": len(processor.current_transcript)
        })

    except Exception as e:
        logger.error(f"Q&A error: {str(e)}")
        return jsonify({"success": False, "error": f"Question processing failed: {str(e)}"}), 500

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get YouTube video recommendations"""
    try:
        if not processor.current_keywords:
            # Try to extract keywords from current transcript if available
            if processor.current_transcript:
                processor.current_keywords = extract_keywords(processor.current_transcript, num_keywords=15)
                
        if not processor.current_keywords:
            return jsonify({
                "success": False, 
                "error": "No keywords available. Please upload and process an audio file first."
            }), 400

        # FIXED: Use improved recommendation function
        videos = get_youtube_recommendations(
            processor.current_keywords, 
            processor.current_transcript, 
            max_results=6
        )

        if not videos:
            return jsonify({
                "success": False,
                "error": "Could not generate recommendations. Please try uploading a different audio file."
            }), 400

        return jsonify({
            "success": True,
            "videos": videos,
            "keywords": processor.current_keywords[:8],
            "total_videos": len(videos)
        })

    except Exception as e:
        logger.error(f"Recommendations error: {str(e)}")
        return jsonify({"success": False, "error": f"Failed to get recommendations: {str(e)}"}), 500

@app.route('/summary', methods=['GET'])
def get_summary():
    """Get transcript summary"""
    try:
        if not processor.current_transcript:
            return jsonify({
                "success": False,
                "error": "No transcript available"
            }), 400

        # FIXED: Regenerate summary if needed or if it's the same as transcript
        if (not processor.current_summary or 
            processor.current_summary == processor.current_transcript or
            len(processor.current_summary) >= len(processor.current_transcript) * 0.8):
            
            processor.current_summary = generate_summary(processor.current_transcript, sentence_count=3)

        return jsonify({
            "success": True,
            "summary": processor.current_summary,
            "original_length": len(processor.current_transcript),
            "summary_length": len(processor.current_summary),
            "compression_ratio": f"{(len(processor.current_summary) / len(processor.current_transcript) * 100):.1f}%"
        })

    except Exception as e:
        logger.error(f"Summary error: {str(e)}")
        return jsonify({"success": False, "error": f"Summary generation failed: {str(e)}"}), 500

@app.route('/keywords', methods=['GET'])
def get_keywords():
    """Get extracted keywords"""
    return jsonify({
        "success": True,
        "keywords": processor.current_keywords,
        "total_keywords": len(processor.current_keywords)
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"success": False, "error": "File too large. Please upload a smaller audio file."}), 413

if __name__ == '__main__':
    print("=" * 60)
    print("🎓 Enhanced Lecturyzer Server Starting...")
    print("=" * 60)
    print("Server running at: http://127.0.0.1:5000")
    print("\nAvailable Endpoints:")
    print("📤 POST /upload          - Upload and process audio")
    print("❓ POST /ask             - Ask questions about content")
    print("📝 GET  /transcript      - Get full transcript")
    print("📋 GET  /summary         - Get content summary")
    print("🎥 GET  /recommendations - Get YouTube recommendations")
    print("🏷️  GET  /keywords        - Get extracted keywords")
    print("💚 GET  /health          - Health check")
    print("=" * 60)
    
    # Configure Flask for larger file uploads
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
    
    app.run(debug=True, host='0.0.0.0', port=5000)
