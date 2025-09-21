import requests
import json
import re
from urllib.parse import quote
import random
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        print(f"NLTK download warning: {e}")

download_nltk_data()

# YouTube API configuration
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"  # Replace with your actual API key
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

# Alternative: Use web scraping approach if no API key
USE_API = False  # Set to True when you have a valid API key

def get_topic_videos(topic, max_results=6):
    """
    Get YouTube videos related to a specific topic with enhanced matching and filtering
    """
    try:
        if not is_valid_topic(topic):
            print(f"Skipping generic topic: '{topic}'")
            return []
            
        if USE_API and YOUTUBE_API_KEY != "YOUR_YOUTUBE_API_KEY":
            return get_videos_via_api(topic, max_results)
        else:
            # Use alternative method without API key
            return get_videos_alternative(topic, max_results)
    except Exception as e:
        print(f"Error getting videos for topic '{topic}': {e}")
        return generate_fallback_videos(topic)

def is_valid_topic(topic):
    """
    Filter out generic phrases and common words that shouldn't be used as search topics
    """
    if not topic or len(topic.strip()) < 3:
        return False
    
    topic_lower = topic.lower().strip()
    
    # Filter out common greetings and generic phrases
    generic_phrases = [
        'hello everyone', 'hello everybody', 'good morning', 'good afternoon', 'good evening',
        'welcome', 'thank you', 'thanks', 'please', 'okay', 'alright', 'well',
        'so', 'now', 'today', 'yesterday', 'tomorrow', 'here', 'there',
        'this is', 'that is', 'we are', 'you are', 'i am', 'let me',
        'going to', 'want to', 'need to', 'have to', 'able to',
        'first', 'second', 'third', 'next', 'then', 'finally',
        'example', 'for example', 'such as', 'like this', 'like that',
        'very good', 'very well', 'very nice', 'really good', 'really well'
    ]
    
    # Check if topic is just a generic phrase
    if topic_lower in generic_phrases:
        return False
    
    # Filter out topics that are too short or just common words
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    words = word_tokenize(topic_lower)
    meaningful_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
    # Need at least one meaningful word
    if len(meaningful_words) == 0:
        return False
    
    # Filter out single common words
    common_words = [
        'people', 'person', 'thing', 'things', 'way', 'ways', 'time', 'times',
        'day', 'days', 'year', 'years', 'work', 'working', 'make', 'making',
        'get', 'getting', 'take', 'taking', 'come', 'coming', 'go', 'going',
        'see', 'seeing', 'know', 'knowing', 'think', 'thinking', 'feel', 'feeling',
        'look', 'looking', 'find', 'finding', 'use', 'using', 'try', 'trying'
    ]
    
    if len(meaningful_words) == 1 and meaningful_words[0] in common_words:
        return False
    
    educational_indicators = [
        'theory', 'concept', 'principle', 'method', 'technique', 'approach',
        'analysis', 'research', 'study', 'experiment', 'model', 'framework',
        'algorithm', 'formula', 'equation', 'process', 'system', 'structure',
        'development', 'implementation', 'application', 'solution', 'problem'
    ]
    
    # Boost score for educational content
    has_educational_content = any(indicator in topic_lower for indicator in educational_indicators)
    
    # Check topic length and complexity
    if len(topic.split()) >= 2 or has_educational_content:
        return True
    
    # Single word topics need to be substantial
    if len(meaningful_words) == 1:
        word = meaningful_words[0]
        # Allow technical terms, proper nouns, or words longer than 4 characters
        if len(word) > 4 or word[0].isupper() or has_educational_content:
            return True
    
    return False

def extract_meaningful_topics(text, max_topics=5):
    """
    Extract only meaningful, educational topics from text
    """
    try:
        # Use existing topic extraction but with better filtering
        from utils.summarizer import extract_key_topics
        
        raw_topics = extract_key_topics(text, max_topics * 3)  # Get more to filter
        
        # Filter topics using our validation
        valid_topics = []
        for topic in raw_topics:
            if is_valid_topic(topic) and topic not in valid_topics:
                valid_topics.append(topic)
        
        scored_topics = []
        for topic in valid_topics:
            score = calculate_topic_relevance_score(topic, text)
            if score > 0.3:  # Minimum relevance threshold
                scored_topics.append((topic, score))
        
        # Sort by relevance score
        scored_topics.sort(key=lambda x: x[1], reverse=True)
        
        return [topic for topic, score in scored_topics[:max_topics]]
    
    except Exception as e:
        print(f"Error extracting meaningful topics: {e}")
        return []

def calculate_topic_relevance_score(topic, context):
    """
    Calculate how relevant a topic is to the educational content
    """
    try:
        topic_words = set(word_tokenize(topic.lower()))
        context_words = set(word_tokenize(context.lower()))
        
        # Basic word overlap
        overlap = len(topic_words & context_words)
        base_score = overlap / len(topic_words) if topic_words else 0
        
        # Boost for educational keywords
        educational_keywords = [
            'learn', 'study', 'understand', 'explain', 'analyze', 'research',
            'theory', 'concept', 'principle', 'method', 'technique', 'approach',
            'development', 'implementation', 'application', 'solution', 'problem'
        ]
        
        educational_boost = sum(1 for word in topic_words if word in educational_keywords) * 0.2
        
        # Boost for technical terms (capitalized words, longer words)
        technical_boost = sum(0.1 for word in topic.split() if len(word) > 6 or word[0].isupper())
        
        # Penalty for very common words
        common_penalty = sum(0.1 for word in topic_words if word in ['good', 'nice', 'great', 'well', 'really'])
        
        final_score = base_score + educational_boost + technical_boost - common_penalty
        return max(0, min(1, final_score))  # Clamp between 0 and 1
    
    except Exception as e:
        return 0.5  # Default score

def get_videos_via_api(topic, max_results=6):
    """Get videos using YouTube Data API v3 with improved search"""
    try:
        search_query = construct_educational_search_query(topic)
        
        # Search for videos
        search_url = f"{YOUTUBE_API_BASE_URL}/search"
        search_params = {
            'part': 'snippet',
            'q': search_query,
            'type': 'video',
            'maxResults': max_results * 2,  # Get more to filter better
            'order': 'relevance',
            'videoDuration': 'medium',  # Prefer medium-length videos
            'videoDefinition': 'high',
            'key': YOUTUBE_API_KEY
        }
        
        response = requests.get(search_url, params=search_params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            videos = []
            
            for item in data.get('items', []):
                video_id = item['id']['videoId']
                snippet = item['snippet']
                
                relevance_score = calculate_educational_relevance_score(topic, snippet)
                
                if relevance_score < 0.3:
                    continue
                
                video_info = {
                    'title': snippet['title'],
                    'description': snippet['description'][:200] + '...' if len(snippet['description']) > 200 else snippet['description'],
                    'thumbnail': snippet['thumbnails']['medium']['url'],
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'channel': snippet['channelTitle'],
                    'published': snippet['publishedAt'][:10],
                    'relevance_score': relevance_score
                }
                videos.append(video_info)
            
            # Sort by relevance score and return top results
            videos.sort(key=lambda x: x['relevance_score'], reverse=True)
            return videos[:max_results]
        
        else:
            print(f"YouTube API error: {response.status_code}")
            return get_videos_alternative(topic, max_results)
    
    except Exception as e:
        print(f"Error in API method: {e}")
        return get_videos_alternative(topic, max_results)

def construct_educational_search_query(topic):
    """
    Construct an optimized search query focused on educational content
    """
    # Clean and enhance the topic
    topic = topic.strip()
    
    # Add educational keywords to improve relevance and filter out casual content
    educational_keywords = ["tutorial", "explained", "course", "lesson", "guide", "lecture", "education"]
    
    # Choose the most appropriate educational keyword based on topic
    if any(word in topic.lower() for word in ['theory', 'concept', 'principle']):
        edu_keyword = "explained"
    elif any(word in topic.lower() for word in ['how to', 'method', 'technique']):
        edu_keyword = "tutorial"
    elif any(word in topic.lower() for word in ['study', 'research', 'analysis']):
        edu_keyword = "lecture"
    else:
        edu_keyword = random.choice(educational_keywords)
    
    # Construct query with educational focus
    enhanced_query = f'"{topic}" {edu_keyword} -vlog -reaction -funny -meme'
    
    return enhanced_query

def calculate_educational_relevance_score(topic, snippet):
    """
    Calculate how relevant and educational a video is to the topic
    """
    score = 0
    topic_words = set(re.findall(r'\b\w+\b', topic.lower()))
    
    # Check title relevance (higher weight)
    title_words = set(re.findall(r'\b\w+\b', snippet['title'].lower()))
    title_overlap = len(topic_words & title_words)
    score += title_overlap * 3
    
    # Check description relevance
    desc_words = set(re.findall(r'\b\w+\b', snippet['description'].lower()))
    desc_overlap = len(topic_words & desc_words)
    score += desc_overlap * 1
    
    # Boost educational channels
    educational_channels = [
        'khan academy', 'coursera', 'edx', 'mit', 'stanford', 'harvard',
        'crash course', 'ted-ed', 'academy', 'university', 'college',
        'education', 'learning', 'tutorial', 'teach', 'professor'
    ]
    channel_name = snippet['channelTitle'].lower()
    if any(edu in channel_name for edu in educational_channels):
        score += 10
    
    # Boost videos with educational keywords in title
    educational_keywords = [
        'tutorial', 'explained', 'course', 'lesson', 'guide', 'introduction',
        'lecture', 'class', 'learn', 'study', 'understand', 'master',
        'fundamentals', 'basics', 'advanced', 'complete guide'
    ]
    title_lower = snippet['title'].lower()
    educational_matches = sum(1 for keyword in educational_keywords if keyword in title_lower)
    score += educational_matches * 2
    
    # Penalty for non-educational content
    non_educational_keywords = [
        'funny', 'hilarious', 'reaction', 'vlog', 'prank', 'challenge',
        'meme', 'tiktok', 'shorts', 'compilation', 'fail', 'epic'
    ]
    penalty = sum(1 for keyword in non_educational_keywords if keyword in title_lower)
    score -= penalty * 5
    
    # Normalize score
    max_possible_score = len(topic_words) * 3 + 10 + len(educational_keywords) * 2
    normalized_score = score / max_possible_score if max_possible_score > 0 else 0
    
    return max(0, min(1, normalized_score))

def get_videos_alternative(topic, max_results=6):
    """
    Alternative method to get video recommendations without API key
    Uses a combination of educational video databases and mock data
    """
    try:
        # Enhanced topic processing
        processed_topic = process_topic_for_search(topic)
        
        # Try to get real videos using web scraping (simplified approach)
        videos = []
        
        # Method 1: Use Invidious instances (YouTube alternative frontends)
        try:
            videos.extend(search_via_invidious(processed_topic, max_results // 2))
        except Exception as e:
            print(f"Invidious search failed: {e}")
        
        # Method 2: Use educational video databases
        try:
            videos.extend(search_educational_databases(processed_topic, max_results // 2))
        except Exception as e:
            print(f"Educational database search failed: {e}")
        
        # If we have some results, return them
        if videos:
            return videos[:max_results]
        
        # Fallback: Generate contextually relevant mock videos
        return generate_contextual_videos(topic, max_results)
    
    except Exception as e:
        print(f"Error in alternative method: {e}")
        return generate_fallback_videos(topic)

def search_via_invidious(topic, max_results=3):
    """Search using Invidious instances (YouTube alternative frontends)"""
    videos = []
    
    # List of public Invidious instances
    invidious_instances = [
        "https://invidious.io",
        "https://invidious.snopyta.org",
        "https://invidious.kavin.rocks"
    ]
    
    for instance in invidious_instances:
        try:
            search_url = f"{instance}/api/v1/search"
            params = {
                'q': topic,
                'type': 'video',
                'sort_by': 'relevance',
                'duration': 'medium'
            }
            
            response = requests.get(search_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data[:max_results]:
                    video_info = {
                        'title': item.get('title', 'Educational Video'),
                        'description': item.get('description', f'Learn about {topic}')[:200] + '...',
                        'thumbnail': f"https://img.youtube.com/vi/{item.get('videoId', '')}/mqdefault.jpg",
                        'url': f"https://www.youtube.com/watch?v={item.get('videoId', '')}",
                        'channel': item.get('author', 'Educational Channel'),
                        'published': item.get('published', '2024-01-01'),
                        'relevance_score': 0.8
                    }
                    videos.append(video_info)
                
                if videos:
                    break  # Stop if we got results from this instance
        
        except Exception as e:
            print(f"Error with instance {instance}: {e}")
            continue
    
    return videos

def search_educational_databases(topic, max_results=3):
    """Search educational video databases and repositories"""
    videos = []
    
    # Mock educational video database (replace with actual API calls)
    educational_sources = [
        {
            'name': 'Khan Academy',
            'base_url': 'https://www.khanacademy.org',
            'subjects': ['mathematics', 'science', 'computer science', 'economics', 'history']
        },
        {
            'name': 'Coursera',
            'base_url': 'https://www.coursera.org',
            'subjects': ['business', 'technology', 'data science', 'arts', 'health']
        },
        {
            'name': 'edX',
            'base_url': 'https://www.edx.org',
            'subjects': ['engineering', 'computer science', 'business', 'humanities']
        }
    ]
    
    # Generate contextually appropriate videos
    topic_lower = topic.lower()
    for source in educational_sources:
        if any(subject in topic_lower for subject in source['subjects']):
            video_info = {
                'title': f"{topic} - {source['name']} Course",
                'description': f"Comprehensive {topic} course from {source['name']}. Learn fundamental concepts and practical applications.",
                'thumbnail': f"https://via.placeholder.com/320x180/4285f4/ffffff?text={source['name']}",
                'url': f"{source['base_url']}/search?query={quote(topic)}",
                'channel': source['name'],
                'published': '2024-01-15',
                'relevance_score': 0.9
            }
            videos.append(video_info)
            
            if len(videos) >= max_results:
                break
    
    return videos

def generate_contextual_videos(topic, max_results=6):
    """Generate contextually relevant video recommendations"""
    videos = []
    
    # Enhanced topic analysis
    topic_keywords = extract_topic_keywords(topic)
    topic_category = categorize_topic(topic)
    
    # Templates for different topic categories
    templates = {
        'science': [
            "{topic} Explained - Scientific Fundamentals",
            "Introduction to {topic} - Lab Demonstration",
            "{topic} in Action - Real World Applications",
            "Advanced {topic} Concepts and Theories"
        ],
        'technology': [
            "{topic} Tutorial - Beginner to Advanced",
            "Mastering {topic} - Best Practices",
            "{topic} Deep Dive - Technical Implementation",
            "Future of {topic} - Industry Insights"
        ],
        'business': [
            "{topic} Strategy - Business Fundamentals",
            "{topic} Case Study - Success Stories",
            "Implementing {topic} in Your Business",
            "{topic} Trends and Market Analysis"
        ],
        'humanities': [
            "Understanding {topic} - Historical Context",
            "{topic} Analysis - Critical Perspectives",
            "{topic} in Modern Society",
            "Exploring {topic} - Cultural Impact"
        ],
        'general': [
            "Complete Guide to {topic}",
            "{topic} Masterclass - Expert Insights",
            "Everything You Need to Know About {topic}",
            "{topic} Simplified - Easy Explanations"
        ]
    }
    
    # Select appropriate templates
    selected_templates = templates.get(topic_category, templates['general'])
    
    # Generate videos
    channels = [
        "EduTech Academy", "Learning Hub", "Knowledge Base", "Expert Tutorials",
        "Academic Insights", "Professional Development", "Study Masters", "Skill Builder"
    ]
    
    for i, template in enumerate(selected_templates[:max_results]):
        title = template.format(topic=topic)
        channel = random.choice(channels)
        
        # Create contextually relevant description
        description = generate_video_description(topic, topic_keywords, topic_category)
        
        video_info = {
            'title': title,
            'description': description,
            'thumbnail': f"https://via.placeholder.com/320x180/ff6b6b/ffffff?text={quote(topic[:20])}",
            'url': f"https://www.youtube.com/results?search_query={quote(title)}",
            'channel': channel,
            'published': f"2024-0{random.randint(1, 9)}-{random.randint(10, 28):02d}",
            'relevance_score': 0.7 + (i * 0.05)  # Slightly decreasing relevance
        }
        videos.append(video_info)
    
    return videos

def generate_fallback_videos(topic, max_results=6):
    """Generate basic fallback videos when all else fails"""
    videos = []
    
    basic_templates = [
        f"Learn {topic} - Comprehensive Guide",
        f"{topic} Fundamentals - Step by Step",
        f"Master {topic} - Professional Course",
        f"{topic} Explained - Easy Tutorial",
        f"Advanced {topic} - Expert Level",
        f"{topic} for Beginners - Start Here"
    ]
    
    for i, title in enumerate(basic_templates[:max_results]):
        video_info = {
            'title': title,
            'description': f"Educational content about {topic}. Learn key concepts and practical applications.",
            'thumbnail': f"https://via.placeholder.com/320x180/007bff/ffffff?text=Video+{i+1}",
            'url': f"https://www.youtube.com/results?search_query={quote(title)}",
            'channel': "Educational Content",
            'published': "2024-01-01",
            'relevance_score': 0.5
        }
        videos.append(video_info)
    
    return videos

def construct_search_query(topic):
    """Construct an optimized search query for the topic"""
    # Clean and enhance the topic
    topic = topic.strip()
    
    # Add educational keywords to improve relevance
    educational_keywords = ["tutorial", "explained", "course", "lesson", "guide"]
    
    # Randomly add one educational keyword
    enhanced_query = f"{topic} {random.choice(educational_keywords)}"
    
    return enhanced_query

def calculate_relevance_score(topic, snippet):
    """Calculate how relevant a video is to the topic"""
    score = 0
    topic_words = set(re.findall(r'\b\w+\b', topic.lower()))
    
    # Check title relevance
    title_words = set(re.findall(r'\b\w+\b', snippet['title'].lower()))
    title_overlap = len(topic_words & title_words)
    score += title_overlap * 2
    
    # Check description relevance
    desc_words = set(re.findall(r'\b\w+\b', snippet['description'].lower()))
    desc_overlap = len(topic_words & desc_words)
    score += desc_overlap
    
    # Boost educational channels
    educational_channels = ['khan academy', 'coursera', 'edx', 'mit', 'stanford', 'harvard']
    if any(edu in snippet['channelTitle'].lower() for edu in educational_channels):
        score += 5
    
    # Boost videos with educational keywords in title
    educational_keywords = ['tutorial', 'explained', 'course', 'lesson', 'guide', 'introduction']
    if any(keyword in snippet['title'].lower() for keyword in educational_keywords):
        score += 3
    
    return score

def process_topic_for_search(topic):
    """Process topic to make it more searchable"""
    # Remove special characters and normalize
    processed = re.sub(r'[^\w\s]', ' ', topic)
    processed = ' '.join(processed.split())  # Normalize whitespace
    
    # Add context for better search results
    if len(processed.split()) == 1:
        processed += " tutorial explanation"
    
    return processed

def extract_topic_keywords(topic):
    """Extract key words from the topic for context"""
    words = re.findall(r'\b\w+\b', topic.lower())
    # Filter out common stop words
    stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords

def categorize_topic(topic):
    """Categorize the topic to select appropriate templates"""
    topic_lower = topic.lower()
    
    science_keywords = ['biology', 'chemistry', 'physics', 'mathematics', 'science', 'research', 'experiment']
    tech_keywords = ['programming', 'software', 'computer', 'technology', 'coding', 'development', 'algorithm']
    business_keywords = ['marketing', 'business', 'management', 'strategy', 'finance', 'economics', 'entrepreneurship']
    humanities_keywords = ['history', 'literature', 'philosophy', 'art', 'culture', 'language', 'sociology']
    
    if any(keyword in topic_lower for keyword in science_keywords):
        return 'science'
    elif any(keyword in topic_lower for keyword in tech_keywords):
        return 'technology'
    elif any(keyword in topic_lower for keyword in business_keywords):
        return 'business'
    elif any(keyword in topic_lower for keyword in humanities_keywords):
        return 'humanities'
    else:
        return 'general'

def generate_video_description(topic, keywords, category):
    """Generate a contextually relevant video description"""
    base_descriptions = {
        'science': f"Explore the fascinating world of {topic}. This comprehensive video covers fundamental principles, latest research findings, and practical applications in real-world scenarios.",
        'technology': f"Master {topic} with this in-depth tutorial. Learn best practices, implementation strategies, and advanced techniques from industry experts.",
        'business': f"Discover how {topic} can transform your business strategy. Gain insights from successful case studies and learn practical implementation methods.",
        'humanities': f"Delve into the rich history and cultural significance of {topic}. Analyze different perspectives and understand its impact on modern society.",
        'general': f"Get a complete understanding of {topic} with this educational video. Perfect for students, professionals, and anyone looking to expand their knowledge."
    }
    
    base_desc = base_descriptions.get(category, base_descriptions['general'])
    
    # Add keywords for better context
    if keywords:
        keyword_str = ', '.join(keywords[:3])
        base_desc += f" Key topics include: {keyword_str}."
    
    return base_desc
