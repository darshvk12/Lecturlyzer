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
    """Generate a comprehensive summary of the text that's distinct from the transcript"""
    try:
        load_models()
        
        if not text or len(text.strip()) < 50:
            return "Text is too short to summarize effectively."
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        if summarizer_model is None:
            return create_enhanced_extractive_summary(text)
        
        # Calculate appropriate max_length based on input length for more concise summaries
        input_length = len(text.split())
        # Make summaries more concise - aim for 10-15% of original length
        max_length = min(150, max(40, input_length // 8))  # More aggressive compression
        min_length = min(25, max_length // 3)
        
        # For very long texts, chunk them
        if len(text) > 1024:
            chunks = chunk_text(text, 800)  # Smaller chunks for better processing
            summaries = []
            
            for chunk in chunks:
                if len(chunk) > 50:  # Skip very short chunks
                    try:
                        chunk_length = len(chunk.split())
                        chunk_max = min(100, max(25, chunk_length // 6))  # More aggressive
                        chunk_min = min(15, chunk_max // 3)
                        
                        summary = summarizer_model(
                            chunk,
                            max_length=chunk_max,
                            min_length=chunk_min,
                            do_sample=False,
                            num_beams=4,
                            length_penalty=2.0,
                            early_stopping=True
                        )
                        summaries.append(summary[0]['summary_text'])
                    except Exception as e:
                        print(f"Error summarizing chunk: {e}")
                        continue
            
            if summaries:
                # Combine chunk summaries and create a final coherent summary
                combined_summary = " ".join(summaries)
                
                # If combined summary is still long, summarize it again
                if len(combined_summary) > 800:
                    try:
                        combined_length = len(combined_summary.split())
                        final_max = min(150, max(40, combined_length // 4))
                        final_min = min(25, final_max // 3)
                        
                        final_summary = summarizer_model(
                            combined_summary,
                            max_length=final_max,
                            min_length=final_min,
                            do_sample=False,
                            num_beams=4,
                            length_penalty=2.0,
                            early_stopping=True
                        )
                        return post_process_summary(final_summary[0]['summary_text'])
                    except:
                        return post_process_summary(combined_summary[:400] + "...")
                
                return post_process_summary(combined_summary)
            else:
                return create_enhanced_extractive_summary(text)
        
        else:
            # Direct summarization for shorter texts
            try:
                summary = summarizer_model(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
                return post_process_summary(summary[0]['summary_text'])
            except Exception as e:
                print(f"Error in direct summarization: {e}")
                return create_enhanced_extractive_summary(text)
    
    except Exception as e:
        print(f"Error in summarization: {e}")
        return create_enhanced_extractive_summary(text)

def post_process_summary(summary):
    """
    Post-process the summary to make it more readable and structured
    """
    try:
        # Clean up the summary
        summary = summary.strip()
        
        # Ensure proper sentence structure
        sentences = sent_tokenize(summary)
        
        # Remove very short sentences (likely fragments)
        meaningful_sentences = [s for s in sentences if len(s.split()) > 3]
        
        # Limit to most important sentences (3-5 for readability)
        if len(meaningful_sentences) > 5:
            meaningful_sentences = meaningful_sentences[:5]
        
        # Join sentences and ensure proper formatting
        processed_summary = " ".join(meaningful_sentences)
        
        # Add structure indicators if the summary covers multiple topics
        if len(meaningful_sentences) >= 3:
            # Try to identify if there are distinct topics
            topics = identify_summary_topics(processed_summary)
            if len(topics) > 1:
                processed_summary = structure_multi_topic_summary(processed_summary, topics)
        
        return processed_summary
    
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return summary

def identify_summary_topics(summary):
    """
    Identify distinct topics in the summary
    """
    try:
        sentences = sent_tokenize(summary)
        topics = []
        
        # Look for topic indicators
        topic_indicators = [
            'first', 'second', 'third', 'next', 'then', 'also', 'additionally',
            'furthermore', 'moreover', 'however', 'in contrast', 'on the other hand'
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in topic_indicators):
                topics.append(sentence)
        
        return topics
    except:
        return []

def structure_multi_topic_summary(summary, topics):
    """
    Add structure to multi-topic summaries
    """
    try:
        # For now, just ensure good flow between sentences
        sentences = sent_tokenize(summary)
        
        # Add transition words where appropriate
        structured_sentences = []
        for i, sentence in enumerate(sentences):
            if i == 0:
                structured_sentences.append(sentence)
            elif i == len(sentences) - 1:
                # Last sentence - add conclusion indicator if not present
                if not any(word in sentence.lower() for word in ['conclusion', 'summary', 'overall', 'finally']):
                    structured_sentences.append(f"Overall, {sentence.lower()}")
                else:
                    structured_sentences.append(sentence)
            else:
                structured_sentences.append(sentence)
        
        return " ".join(structured_sentences)
    except:
        return summary

def create_enhanced_extractive_summary(text, num_sentences=4):
    """Create an enhanced extractive summary by selecting important sentences with better scoring"""
    try:
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        sentence_scores = {}
        
        # Get word frequencies
        try:
            words = word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = text.lower().split()
        
        # Filter meaningful words
        meaningful_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        word_freq = Counter(meaningful_words)
        
        # Score sentences based on multiple factors
        for i, sentence in enumerate(sentences):
            try:
                sentence_words = word_tokenize(sentence.lower())
            except:
                sentence_words = sentence.lower().split()
            
            score = 0
            word_count = 0
            
            # Factor 1: Word frequency score
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            # Factor 2: Position score (first and last sentences are often important)
            position_score = 0
            if i == 0:  # First sentence
                position_score = 2
            elif i == len(sentences) - 1:  # Last sentence
                position_score = 1.5
            elif i < len(sentences) * 0.3:  # Early sentences
                position_score = 1.2
            
            # Factor 3: Sentence length score (prefer medium-length sentences)
            length_score = 0
            sentence_length = len(sentence_words)
            if 10 <= sentence_length <= 25:  # Optimal length
                length_score = 1.5
            elif 6 <= sentence_length <= 30:  # Good length
                length_score = 1.0
            else:
                length_score = 0.5
            
            # Factor 4: Keyword presence (educational terms, important concepts)
            keyword_score = 0
            educational_keywords = [
                'important', 'key', 'main', 'primary', 'significant', 'crucial',
                'essential', 'fundamental', 'basic', 'concept', 'principle',
                'theory', 'method', 'approach', 'result', 'conclusion'
            ]
            
            sentence_lower = sentence.lower()
            for keyword in educational_keywords:
                if keyword in sentence_lower:
                    keyword_score += 0.5
            
            # Combine all factors
            if word_count > 0:
                final_score = (score / word_count) + position_score + length_score + keyword_score
                sentence_scores[sentence] = final_score
        
        # Select top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        
        # Maintain original order for coherence
        summary_sentences = []
        for sentence in sentences:
            if any(sentence == s[0] for s in top_sentences):
                summary_sentences.append(sentence)
        
        # Create a more structured summary
        summary_text = " ".join(summary_sentences)
        
        # Add summary structure
        if len(summary_sentences) > 2:
            # Try to identify the main topic from the first sentence
            first_sentence = summary_sentences[0]
            if not first_sentence.lower().startswith(('this', 'the lecture', 'in this')):
                summary_text = f"This lecture covers {first_sentence.lower()}"
                if len(summary_sentences) > 1:
                    summary_text += f" {' '.join(summary_sentences[1:])}"
        
        return summary_text
    
    except Exception as e:
        print(f"Error in enhanced extractive summary: {e}")
        return text[:400] + "..." if len(text) > 400 else text

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
