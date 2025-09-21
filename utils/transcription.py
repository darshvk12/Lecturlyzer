import whisper
import os
from pydub import AudioSegment
import tempfile

# Load Whisper model globally to avoid reloading
model = None

def load_model():
    global model
    if model is None:
        print("Loading Whisper model...")
        try:
            # Start with base model for better speed/accuracy balance
            model = whisper.load_model("base")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            # Fallback to tiny model if base fails
            try:
                model = whisper.load_model("tiny")
                print("Loaded tiny Whisper model as fallback")
            except Exception as e2:
                print(f"Failed to load any Whisper model: {e2}")
                raise e2

def convert_to_wav(input_path):
    """Convert audio file to WAV format for better compatibility"""
    try:
        # Load audio file
        audio = AudioSegment.from_file(input_path)
        
        # Convert to mono and set sample rate to 16kHz (optimal for Whisper)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            audio.export(temp_path, format='wav')
            return temp_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return input_path  # Return original if conversion fails

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using Whisper
    """
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load model if not already loaded
        load_model()
        
        print(f"Starting transcription of: {audio_path}")
        
        # Convert to WAV for better compatibility
        wav_path = convert_to_wav(audio_path)
        temp_file_created = wav_path != audio_path
        
        try:
            # Transcribe with Whisper
            result = model.transcribe(
                wav_path,
                language='en',  # Specify English for better performance
                task='transcribe',
                fp16=False,  # Use fp32 for better compatibility
                verbose=False
            )
            
            transcript = result['text'].strip()
            
            if not transcript:
                raise ValueError("Transcription resulted in empty text")
            
            print(f"Transcription completed. Length: {len(transcript)} characters")
            return transcript
            
        finally:
            # Clean up temporary WAV file
            if temp_file_created and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except:
                    pass
    
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        # Return a fallback message instead of None
        return f"Transcription failed: {str(e)}. Please try uploading a different audio file."

def chunk_long_audio(audio_path, chunk_length_ms=60000):
    """
    Split long audio files into chunks for better processing
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        chunks = []
        
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i + chunk_length_ms]
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                chunk.export(temp_file.name, format='wav')
                chunks.append(temp_file.name)
        
        return chunks
    except Exception as e:
        print(f"Error chunking audio: {e}")
        return [audio_path]

def transcribe_long_audio(audio_path):
    """
    Transcribe long audio files by chunking them
    """
    try:
        # Check file size
        file_size = os.path.getsize(audio_path)
        
        # If file is larger than 25MB, chunk it
        if file_size > 25 * 1024 * 1024:
            print("Large file detected, chunking for processing...")
            chunks = chunk_long_audio(audio_path)
            
            full_transcript = ""
            for i, chunk_path in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_transcript = transcribe_audio(chunk_path)
                if chunk_transcript and "Transcription failed" not in chunk_transcript:
                    full_transcript += chunk_transcript + " "
                
                # Clean up chunk file
                try:
                    os.unlink(chunk_path)
                except:
                    pass
            
            return full_transcript.strip()
        else:
            return transcribe_audio(audio_path)
    
    except Exception as e:
        print(f"Error in long audio transcription: {e}")
        return transcribe_audio(audio_path)
