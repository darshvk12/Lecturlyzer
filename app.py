# 1. Install Whisper and Transformers
!pip install openai-whisper
!pip install transformers --quiet

# 2. Import necessary libraries
import whisper
from transformers import pipeline

# 3. Transcribe audio
model = whisper.load_model("base")
result = model.transcribe("Recording.m4a")  # Make sure your file is uploaded
transcript = result["text"]
print("Transcript:\n", transcript)

# 4. Summarize (Optional)
summarizer = pipeline("summarization")
text = transcript[:1000]  # Truncate for model limit
summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
print("\nSummary:\n", summary)

# 5. QA System
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def ask_question(question):
    result = qa_pipeline(question=question, context=transcript)
    return result["answer"]

# 6. Interactive Q&A
while True:
    user_question = input("\nAsk a question (or type 'exit'): ")
    if user_question.lower() == 'exit':
        break
    answer = ask_question(user_question)
    print("Answer:", answer)
