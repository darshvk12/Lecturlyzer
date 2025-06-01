# 📚 Lecturyzer

Lecturyzer is an AI-powered web application that transforms lecture audio into comprehensive learning resources. It transcribes lectures, summarizes content, highlights key points, generates an index of topics, answers user queries, and even recommends related YouTube videos — all in one intuitive platform.

## 🔍 Features

- 🎙️ Upload Audio – Upload lectures in .mp3, .wav, or other audio formats.
- 📝 Transcription – Generate accurate text transcripts from lecture audio.
- ✨ Summary – Get concise summaries of lengthy lectures.
- 📌 Key Points – Extract the most important points and insights.
- ❓ Question Answering – Ask questions and get AI-generated answers based on the lecture.
- 📺 YouTube Suggestions – Receive video recommendations related to lecture content.

🔗 Live Demo (Add your deployment link)  
📹 Demo Video (Add your YouTube link if available)

## 🛠️ Tech Stack

Frontend: HTML5, CSS3, JavaScript 
Backend: Python (Flask), SpeechRecognition, Google Translate API, gTTS, HuggingFace Transformers. 
Others: YouTube Data API.

## 📂 Project Structure

Lecturyzer/  
├── app.py                  # Flask backend  
├── static/                 # CSS, JS, images  
├── templates/              # HTML templates  
├── uploads/                # Uploaded audio files  
├── utils/  
│   ├── transcription.py    # Transcription logic  
│   ├── summarizer.py       # Summarization logic  
│   ├── question_answer.py  # Q&A system  
│   ├── youtube_recommend.py # YouTube API integration  
├── requirements.txt        # Dependencies  
└── README.md               # Project documentation

## ⚙️ Installation & Setup

1. Clone the repository  
   git clone https://github.com/darshvk12/lecturyzer.git  
   cd lecturyzer

2. Create a virtual environment  
   python -m venv venv  
   venv\Scripts\activate on Windows

3. Install dependencies  
   pip install -r requirements.txt

4. Set up environment variables  
   Create a `.env` file with your API keys:  
   OPENAI_API_KEY=your_api_key  
   YOUTUBE_API_KEY=your_api_key

5. Run the app  
   python app.py

6. Open in browser  
   http://127.0.0.1:5000/


## 🧠 How It Works

1. Audio is uploaded and processed using SpeechRecognition or Whisper to generate text.
2. The transcript is summarized using transformer models (e.g., T5, BART, or OpenAI/Gemini APIs).
3. Key points are extracted using NLP techniques.
4. A Q&A system answers user questions based on the lecture transcript.
5. Topics are identified and passed to the YouTube API to recommend relevant educational videos.

## 🔐 Security & Limitations

- API rate limits may apply (OpenAI, YouTube).
- Transcription accuracy depends on audio clarity.
- Data persistence is temporary by default (can be extended).

## 📈 Future Enhancements

- PDF export of transcripts and summaries  
- Voice-based chatbot for Q&A  
- Personalized content recommendations  
- User accounts with history and analytics

## 🤝 Contributing

Contributions are welcome!  
Fork this repository, create a new branch, make your changes, and open a pull request.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙋‍♂️ Author

Developed by Darsh  
📧 darshvk12@gmail.com  

## ⭐ Show Your Support

If you like this project, give it a ⭐ on GitHub and share it with your friends!
