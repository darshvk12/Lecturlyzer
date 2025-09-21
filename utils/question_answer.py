from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import nltk
from nltk.tokenize import sent_tokenize
import re
from collections import Counter
import random
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
    except Exception as e:
        print(f"NLTK download warning: {e}")

download_nltk_data()

# Global models
qa_model = None
question_generator = None

def load_models():
    global qa_model, question_generator
    
    if qa_model is None:
        try:
            print("Loading Q&A model...")
            qa_model = pipeline(
                "question-answering",
                model="distilbert-base-cased-distilled-squad",
                tokenizer="distilbert-base-cased-distilled-squad"
            )
            print("Q&A model loaded successfully")
        except Exception as e:
            print(f"Error loading Q&A model: {e}")
            try:
                # Fallback to a different model
                qa_model = pipeline("question-answering")
                print("Loaded fallback Q&A model")
            except Exception as e2:
                print(f"Failed to load any Q&A model: {e2}")

def chunk_text_for_qa(text, max_length=512):
    """Split text into overlapping chunks for Q&A"""
    try:
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        overlap_sentences = 2  # Number of sentences to overlap
        
        i = 0
        while i < len(sentences):
            current_chunk = ""
            sentence_count = 0
            
            # Add sentences to current chunk
            j = max(0, i - overlap_sentences) if i > 0 else i
            while j < len(sentences) and len(current_chunk) < max_length:
                current_chunk += sentences[j] + " "
                if j >= i:  # Only count new sentences
                    sentence_count += 1
                j += 1
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Move to next chunk
            i += max(1, sentence_count - overlap_sentences)
        
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        return [text]

def answer_question(question, context):
    """Answer a question based on the given context"""
    try:
        load_models()
        
        if not question or not context:
            return "Please provide both a question and context."
        
        if qa_model is None:
            return answer_question_simple(question, context)
        
        # Clean inputs
        question = question.strip()
        context = re.sub(r'\s+', ' ', context).strip()
        
        # If context is too long, chunk it and find the best answer
        if len(context) > 512:
            chunks = chunk_text_for_qa(context)
            best_answer = ""
            best_score = 0
            
            for chunk in chunks:
                try:
                    result = qa_model(question=question, context=chunk)
                    if result['score'] > best_score:
                        best_score = result['score']
                        best_answer = result['answer']
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    continue
            
            if best_answer and best_score > 0.1:  # Minimum confidence threshold
                return f"{best_answer}\n\n(Confidence: {best_score:.2f})"
            else:
                return answer_question_simple(question, context)
        
        else:
            # Direct Q&A for shorter contexts
            try:
                result = qa_model(question=question, context=context)
                if result['score'] > 0.1:
                    return f"{result['answer']}\n\n(Confidence: {result['score']:.2f})"
                else:
                    return answer_question_simple(question, context)
            except Exception as e:
                print(f"Error in direct Q&A: {e}")
                return answer_question_simple(question, context)
    
    except Exception as e:
        print(f"Error answering question: {e}")
        return answer_question_simple(question, context)

def answer_question_simple(question, context):
    """Simple keyword-based question answering as fallback"""
    try:
        question_lower = question.lower()
        try:
            sentences = sent_tokenize(context)
        except:
            sentences = context.split('.')
        
        # Extract keywords from question
        question_words = re.findall(r'\b\w+\b', question_lower)
        question_words = [word for word in question_words if len(word) > 3]
        
        # Score sentences based on keyword overlap
        sentence_scores = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = sum(1 for word in question_words if word in sentence_lower)
            if score > 0:
                sentence_scores.append((sentence, score))
        
        if sentence_scores:
            # Sort by score and return the best sentence(s)
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 1-2 sentences
            best_sentences = [s[0] for s in sentence_scores[:2]]
            return " ".join(best_sentences)
        else:
            return "I couldn't find a specific answer to your question in the lecture content. Please try rephrasing your question or ask about a different topic."
    
    except Exception as e:
        return f"Error processing question: {str(e)}"

def generate_questions(text, num_questions=8):
    """Generate relevant questions from the text"""
    try:
        if not text or len(text.strip()) < 100:
            return ["What is the main topic discussed?", "What are the key points?"]
        
        questions = []
        
        # Method 1: Template-based questions
        template_questions = generate_template_questions(text)
        questions.extend(template_questions)
        
        # Method 2: Content-specific questions
        content_questions = generate_content_questions(text)
        questions.extend(content_questions)
        
        # Remove duplicates and limit to requested number
        unique_questions = []
        seen = set()
        for q in questions:
            q_lower = q.lower().strip()
            if q_lower not in seen and len(q) > 10:
                unique_questions.append(q)
                seen.add(q_lower)
        
        return unique_questions[:num_questions]
    
    except Exception as e:
        print(f"Error generating questions: {e}")
        return [
            "What is the main topic of this lecture?",
            "What are the key concepts discussed?",
            "What examples are provided?",
            "What conclusions are drawn?"
        ]

def generate_template_questions(text):
    """Generate questions using templates"""
    questions = []
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')
    
    # Question templates
    templates = [
        "What is {}?",
        "How does {} work?",
        "Why is {} important?",
        "What are the benefits of {}?",
        "What are the challenges of {}?",
        "How can {} be improved?",
        "What is the purpose of {}?",
        "What are the characteristics of {}?"
    ]
    
    # Extract potential topics for templates
    topics = extract_topics_for_questions(text)
    
    # Generate questions from templates
    for topic in topics[:4]:  # Limit topics
        template = random.choice(templates)
        question = template.format(topic)
        questions.append(question)
    
    return questions

def generate_content_questions(text):
    """Generate questions based on content analysis"""
    questions = []
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('.')
    
    # Look for definition patterns
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Pattern: "X is defined as..." or "X means..."
        if " is " in sentence_lower or " means " in sentence_lower:
            # Extract the subject
            parts = sentence.split(" is ")
            if len(parts) >= 2:
                subject = parts[0].strip()
                if len(subject) < 50 and not subject.lower().startswith(('this', 'that', 'it', 'they')):
                    questions.append(f"What is {subject}?")
        
        # Pattern: "There are X types of..." or "X includes..."
        if "types of" in sentence_lower or "includes" in sentence_lower:
            if "types of" in sentence_lower:
                topic = sentence_lower.split("types of")[1].split()[0:3]
                topic_str = " ".join(topic).strip('.,!?')
                questions.append(f"What are the types of {topic_str}?")
        
        # Pattern: "The process of..." or "The method of..."
        if "process of" in sentence_lower or "method of" in sentence_lower:
            questions.append("What is the process described?")
        
        # Pattern: Numbers and statistics
        if any(char.isdigit() for char in sentence) and ("%" in sentence or "percent" in sentence.lower()):
            questions.append("What are the key statistics mentioned?")
    
    # Look for cause-effect relationships
    cause_effect_words = ['because', 'therefore', 'as a result', 'consequently', 'due to', 'leads to']
    for sentence in sentences:
        if any(word in sentence.lower() for word in cause_effect_words):
            questions.append("What are the cause and effect relationships discussed?")
            break
    
    # Look for comparison patterns
    comparison_words = ['compared to', 'versus', 'different from', 'similar to', 'unlike']
    for sentence in sentences:
        if any(word in sentence.lower() for word in comparison_words):
            questions.append("What comparisons are made in the lecture?")
            break
    
    # Look for procedural content
    procedural_words = ['first', 'second', 'then', 'next', 'finally', 'step']
    if any(any(word in sentence.lower() for word in procedural_words) for sentence in sentences[:10]):
        questions.append("What are the steps or procedures mentioned?")
    
    # General content questions
    content_questions = [
        "What are the main points discussed in this lecture?",
        "What examples are provided to illustrate the concepts?",
        "What conclusions can be drawn from this lecture?",
        "How does this topic relate to real-world applications?",
        "What are the implications of the concepts discussed?",
        "What questions remain unanswered or need further research?"
    ]
    
    questions.extend(content_questions[:3])  # Add a few general questions
    
    return questions

def extract_topics_for_questions(text):
    """Extract potential topics for question generation"""
    topics = []
    
    # Simple noun extraction
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    
    # Count frequency
    word_count = Counter(words)
    
    # Get most common topics
    common_topics = [word for word, count in word_count.most_common(10) if count > 1]
    
    # Filter out common words
    stop_words = {'The', 'This', 'That', 'These', 'Those', 'There', 'Here', 'Where', 'When', 'What', 'How', 'Why'}
    topics = [topic for topic in common_topics if topic not in stop_words and len(topic) > 2]
    
    if not topics:  # Fallback if no topics found
        topics = ["topic", "concept", "subject", "material"]
    
    return topics[:8]

def get_question_difficulty_level(question, context):
    """Assess the difficulty level of answering a question"""
    try:
        # Simple heuristic based on question type and context complexity
        question_lower = question.lower()
        
        # Easy questions
        if any(starter in question_lower for starter in ['what is', 'who is', 'when is', 'where is']):
            return "Easy"
        
        # Medium questions
        elif any(starter in question_lower for starter in ['how does', 'why does', 'what are']):
            return "Medium"
        
        # Hard questions
        elif any(starter in question_lower for starter in ['analyze', 'evaluate', 'compare', 'contrast']):
            return "Hard"
        
        else:
            return "Medium"
    
    except:
        return "Medium"

def validate_generated_questions(questions, context):
    """Validate that generated questions make sense with the context"""
    valid_questions = []
    
    for question in questions:
        # Basic validation
        if (len(question) > 10 and 
            question.endswith('?') and 
            not question.lower().startswith('what is the the')):  # Avoid malformed questions
            
            # Check if question words are in context (basic relevance check)
            question_words = re.findall(r'\b\w+\b', question.lower())
            context_words = re.findall(r'\b\w+\b', context.lower())
            
            # At least some overlap should exist
            overlap = len(set(question_words) & set(context_words))
            if overlap > 2 or len(question_words) < 6:  # Short questions get a pass
                valid_questions.append(question)
    
    return valid_questions