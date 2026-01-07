# 📚 Lecturyzer

Lecturyzer is an AI-powered web application that transforms lecture audio into comprehensive learning resources. It helps students and educators by automatically transcribing, summarizing, extracting key points, generating an index of topics, answering queries, and recommending YouTube videos — all in one intuitive platform.

# 🚀 Features

🎙️ Upload Audio – Upload lectures in .mp3, .wav, or other formats.

📝 Transcription – Generate accurate text transcripts from lecture audio.

✨ Summarization – Get concise summaries of lengthy lectures.

📌 Key Points Extraction – Highlight the most important points and insights.

❓ Question Answering – Ask questions and get AI-generated answers based on the lecture content.

📺 YouTube Suggestions – Receive educational video recommendations related to lecture topics.

# 🛠️ Tech Stack

Frontend: HTML5, CSS3, JavaScript

Backend: Python (Flask)

APIs & Libraries:

SpeechRecognition / Whisper (transcription)

HuggingFace Transformers (summarization & QnA)

Google Translate API & gTTS (multilingual support)

YouTube Data API (video recommendations)

# 📂 Project Structure
Lecturyzer/

│── app.py                  # Flask backend (main entry point)

│── requirements.txt        # Python dependencies

│── README.md               # Project documentation

│

├── static/                 # CSS, JS, images

│

├── templates/              # HTML templates (frontend)

│ 
└── index.html

│

├── uploads/                # Uploaded audio files
│
└── utils/                  # Core application logic
    ├── transcription.py    # Handles lecture transcription
    ├── summarizer.py       # Summarization logic
    ├── question_answer.py  # Q&A system
    └── youtube_recommend.py# YouTube API integration

# ⚙️ Installation & Setup

Clone the repository

git clone https://github.com/darshvk12/lecturyzer.git
cd lecturyzer


Create a virtual environment

python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


# Install dependencies

pip install -r requirements.txt


Set up environment variables
Create a .env file in the root directory with your API keys:

OPENAI_API_KEY=your_openai_key

Run the app

python app.py


Open in browser

http://127.0.0.1:5000/

#  How It Works

Upload an audio lecture file.

Transcription → Converts audio into text using SpeechRecognition or Whisper.

Summarization & Key Points → Transformer models generate a concise summary and key insights.

Q&A System → Users ask questions, and AI responds based on the transcript.

Topic Extraction & YouTube Suggestions → Identifies core topics and fetches related YouTube videos.

# 🔐 Security & Limitations

API rate limits may apply (OpenAI, YouTube).

Transcription accuracy depends on audio clarity.

Uploaded files and generated content are temporary by default (extendable).

# 📈 Future Enhancements

📄 Export transcripts & summaries as PDF

🎤 Voice-enabled chatbot for Q&A

🎯 Personalized lecture recommendations

👤 User accounts with history & analytics

# 🤝 Contributing

Contributions are always welcome!

Fork this repository

Create a new branch (feature/new-feature)

Commit your changes

Open a Pull Request

# 📜 License

This project is licensed under the MIT License. See the LICENSE
 file for details.

# 🙋‍♂️ Author

Darsh
📧 darshvk12@gmail.com

⭐ Show Your Support

If you found this project helpful, give it a star ⭐ on GitHub and share it with your peers!
