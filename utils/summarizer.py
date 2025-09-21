from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data with error handling
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('wordnet', quiet=True)
    except Exception as e:
        print(f"NLTK download warning: {e}")

download_nltk_data()

# Initialize models
summarizer_model = None
nlp = None

def load_models():
    global summarizer_model, nlp
    
    if summarizer_model is None:
        try:
            print("Loading summarization model...")
            summarizer_model = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
            print("Summarization model loaded successfully")
        except Exception as e:
            print(f"Error loading summarization model: {e}")
            try:
                # Fallback to a smaller model
                summarizer_model = pipeline(
                    "summarization",
                    model="sshleifer/distilbart-cnn-12-6"
                )
                print("Loaded fallback summarization model")
            except Exception as e2:
                print(f"Failed to load any summarization model: {e2}")
    
    if nlp is None:
        try:
            print("Loading spaCy model...")
            import spacy
            nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully")
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")

def chunk_text(text, max_length=1024):
    """Split text into chunks suitable for summarization"""
    try:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        return [text]

def summarize_text(text):
    """Generate a comprehensive summary of the text"""
    try:
        load_models()
        
        if not text or len(text.strip()) < 50:
            return "Text is too short to summarize effectively."
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if summarizer_model is None:
            return create_extractive_summary(text)
        
        # Calculate appropriate max_length based on input length
        input_length = len(text.split())
        max_length = min(200, max(50, input_length // 3))  # Dynamic max length
        min_length = min(30, max_length // 2)
        
        # For very long texts, chunk them
        if len(text) > 1024:
            chunks = chunk_text(text)
            summaries = []
            
            for chunk in chunks:
                if len(chunk) > 50:  # Skip very short chunks
                    try:
                        chunk_length = len(chunk.split())
                        chunk_max = min(150, max(30, chunk_length // 3))
                        chunk_min = min(20, chunk_max // 2)
                        
                        summary = summarizer_model(
                            chunk,
                            max_length=chunk_max,
                            min_length=chunk_min,
                            do_sample=False
                        )
                        summaries.append(summary[0]['summary_text'])
                    except Exception as e:
                        print(f"Error summarizing chunk: {e}")
                        continue
            
            if summaries:
                # Combine chunk summaries
                combined_summary = " ".join(summaries)
                
                # If combined summary is still long, summarize it again
                if len(combined_summary) > 1024:
                    try:
                        combined_length = len(combined_summary.split())
                        final_max = min(200, max(50, combined_length // 3))
                        final_min = min(30, final_max // 2)
                        
                        final_summary = summarizer_model(
                            combined_summary,
                            max_length=final_max,
                            min_length=final_min,
                            do_sample=False
                        )
                        return final_summary[0]['summary_text']
                    except:
                        return combined_summary[:500] + "..."
                
                return combined_summary
            else:
                return create_extractive_summary(text)
        
        else:
            # Direct summarization for shorter texts
            try:
                summary = summarizer_model(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )
                return summary[0]['summary_text']
            except Exception as e:
                print(f"Error in direct summarization: {e}")
                return create_extractive_summary(text)
    
    except Exception as e:
        print(f"Error in summarization: {e}")
        return create_extractive_summary(text)

def create_extractive_summary(text, num_sentences=5):
    """Create an extractive summary by selecting important sentences"""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Calculate sentence scores based on word frequency
        try:
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
        except:
            # Fallback if stopwords not available
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = text.lower().split()
        
        words = [word for word in words if word.isalnum() and word not in stop_words]
        
        word_freq = Counter(words)
        
        sentence_scores = {}
        for sentence in sentences:
            try:
                sentence_words = word_tokenize(sentence.lower())
            except:
                sentence_words = sentence.lower().split()
            
            score = 0
            word_count = 0
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if any(sentence == s[0] for s in top_sentences):
                summary_sentences.append(sentence)
        
        return " ".join(summary_sentences)
    
    except Exception as e:
        print(f"Error in extractive summary: {e}")
        return text[:500] + "..." if len(text) > 500 else text

def extract_key_topics(text, max_topics=10):
    """Extract key topics from the text using multiple methods"""
    try:
        load_models()
        
        topics = set()
        
        # Method 1: Named Entity Recognition with spaCy (if available)
        if nlp is not None:
            try:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'EVENT', 'PRODUCT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']:
                        topic = ent.text.strip()
                        if len(topic) > 2 and len(topic) < 50:
                            topics.add(topic)
            except Exception as e:
                print(f"Error in NER topic extraction: {e}")
        
        # Method 2: Important noun phrases (if spaCy available)
        if nlp is not None:
            try:
                doc = nlp(text)
                for chunk in doc.noun_chunks:
                    if len(chunk.text.strip()) > 3 and len(chunk.text.strip()) < 50:
                        # Filter out common phrases
                        chunk_text = chunk.text.strip().lower()
                        if not any(common in chunk_text for common in ['this', 'that', 'these', 'those', 'some', 'many', 'few']):
                            topics.add(chunk.text.strip())
            except Exception as e:
                print(f"Error in noun phrase extraction: {e}")
        
        # Method 3: POS tagging for important nouns (with error handling)
        try:
            sentences = sent_tokenize(text)
            for sentence in sentences[:10]:  # Focus on first 10 sentences
                try:
                    words = word_tokenize(sentence)
                    pos_tags = pos_tag(words)
                    
                    # Extract consecutive nouns
                    noun_phrase = []
                    for word, pos in pos_tags:
                        if pos.startswith('NN') or pos.startswith('JJ'):  # Nouns and adjectives
                            noun_phrase.append(word)
                        else:
                            if len(noun_phrase) >= 2:
                                phrase = ' '.join(noun_phrase)
                                if 3 < len(phrase) < 50:
                                    topics.add(phrase)
                            noun_phrase = []
                    
                    # Don't forget the last phrase
                    if len(noun_phrase) >= 2:
                        phrase = ' '.join(noun_phrase)
                        if 3 < len(phrase) < 50:
                            topics.add(phrase)
                except Exception as e:
                    print(f"Error processing sentence in POS tagging: {e}")
                    continue
        except Exception as e:
            print(f"Error in POS tagging: {e}")
        
        # Method 4: Frequency-based keyword extraction (always works)
        try:
            # Clean text and extract words
            clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
            try:
                words = word_tokenize(clean_text)
                stop_words = set(stopwords.words('english'))
            except:
                words = clean_text.split()
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            # Remove stopwords and short words
            words = [word for word in words if len(word) > 3 and word not in stop_words]
            
            # Get most frequent words as potential topics
            word_freq = Counter(words)
            frequent_words = [word for word, count in word_freq.most_common(20) if count >= 2]
            
            for word in frequent_words:
                if word.isalpha():  # Only alphabetic words
                    topics.add(word.capitalize())
        
        except Exception as e:
            print(f"Error in frequency-based extraction: {e}")
        
        # Convert to list and sort by relevance (longer phrases first, then alphabetically)
        topic_list = list(topics)
        topic_list.sort(key=lambda x: (-len(x.split()), x.lower()))
        
        # Filter and clean topics
        cleaned_topics = []
        for topic in topic_list[:max_topics * 2]:  # Get more than needed for filtering
            topic = topic.strip()
            if (len(topic) > 2 and 
                not topic.lower() in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'] and
                not re.match(r'^\d+$', topic)):  # Not just numbers
                cleaned_topics.append(topic)
        
        # If we have no topics, create some basic ones
        if not cleaned_topics:
            cleaned_topics = ["General Topic", "Key Concepts", "Learning Material"]
        
        return cleaned_topics[:max_topics]
    
    except Exception as e:
        print(f"Error in topic extraction: {e}")
        # Fallback: simple word frequency
        try:
            try:
                words = word_tokenize(text.lower())
                stop_words = set(stopwords.words('english'))
            except:
                words = text.lower().split()
                stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            
            words = [word for word in words if len(word) > 4 and word.isalpha() and word not in stop_words]
            
            word_freq = Counter(words)
            return [word.capitalize() for word, _ in word_freq.most_common(max_topics)]
        except:
            return ["General Topics", "Education", "Learning"]

def get_text_statistics(text):
    """Get basic statistics about the text"""
    try:
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
        except:
            sentences = text.split('.')
            words = text.split()
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'character_count': len(text),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    except Exception as e:
        return {'error': str(e)}