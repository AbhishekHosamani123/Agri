# 🌾 Agriculture AI Assistant - Enhanced with Gemini

An advanced AI-powered web application for answering questions about Indian agriculture, crop production, and soil health data.

## ✨ Key Features

- 🤖 **Gemini AI Powered**: Enhanced with Google's Gemini AI for natural, conversational responses
- 💬 **Modern Chat Interface**: ChatGPT-style UI with beautiful design
- 📊 **Multi-Dataset Support**: Crop production and soil health data across India
- 🔍 **Intelligent Search**: Vector-based semantic search with TF-IDF
- 📈 **Source Tracking**: Every answer includes citations and confidence scores
- 🎨 **Beautiful UI**: Gradient design with smooth animations

## 🚀 Quick Start

### 1. Install Dependencies

**Backend:**
```bash
cd app/backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd app/frontend
npm install
```

### 2. Start the Application

**Windows (Easy):**
```bash
# Double-click start.bat in app folder
# OR
cd app
start.bat
```

**Manual:**
```bash
# Terminal 1 - Backend
cd app/backend
python app.py

# Terminal 2 - Frontend
cd app/frontend
npm run dev
```

### 3. Access the App
Open: **http://localhost:3000**

## 🎯 Example Questions

- ✅ "What is the rice production in Andhra Pradesh?"
- ✅ "Tell me about cotton production in Maharashtra"
- ✅ "What is the soil pH in Kerala?"
- ✅ "What crops are grown in Punjab?"
- ✅ "Compare wheat production in Punjab and Haryana"
- ✅ "Explain soil health trends in Rajasthan"

## 🏗️ Architecture

```
┌─────────────────┐
│  React Frontend │  (Modern UI with Gradient Design)
│  Port: 3000     │
└────────┬────────┘
         │ HTTP API
         ▼
┌─────────────────┐
│  Flask Backend  │  (REST API + Q&A System)
│  Port: 5000     │
└────────┬────────┘
         │
         ├─► Vector Database (Semantic Search)
         │
         └─► Gemini AI (Enhanced Responses)
```

## 📂 Project Structure

```
app/
├── backend/
│   ├── app.py              # Flask API server
│   ├── qa_system.py        # Q&A system logic
│   ├── gemini_service.py   # Gemini AI integration
│   ├── config.py           # API keys config
│   ├── requirements.txt    # Python dependencies
│   └── vector_database.pkl # Trained model
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   └── Chat.jsx    # Chat interface
    │   ├── App.jsx
    │   └── main.jsx
    ├── package.json
    └── vite.config.js
```

## 🤖 How It Works

1. **User asks a question** in the chat interface
2. **Frontend sends request** to Flask backend
3. **Backend searches** the vector database for relevant data
4. **Gemini AI enhances** the response for better quality
5. **Frontend displays** the natural, conversational answer

## 🎨 UI Features

- **Gradient design** with blue/green theme
- **Shadow effects** for depth
- **Animated loading** indicators
- **Source citations** displayed
- **Confidence scores** visible
- **Responsive sidebar** for chat history
- **Real-time typing** indication

## 🔧 API Endpoints

### POST /query
Query the AI system with a question

**Request:**
```json
{
  "question": "What is rice production in Andhra Pradesh?",
  "use_gemini": true,
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "...",
  "answer": "AI-enhanced natural response...",
  "confidence": 0.45,
  "sources": [...],
  "ai_enhanced": true
}
```

### GET /health
Health check endpoint

### GET /stats
System statistics

## 📊 Tech Stack

- **Backend**: Python, Flask, scikit-learn
- **AI**: Google Gemini Pro
- **Vector DB**: TF-IDF with cosine similarity
- **Frontend**: React, Vite, TailwindCSS, Axios
- **Icons**: Lucide React

## 🎓 What's Different from Basic ChatBot?

### Before (Basic):
- Simple retrieval of data
- Technical, robotic responses
- Basic UI

### After (Enhanced):
- ✨ Natural, conversational AI responses
- 🧠 Context-aware answers
- 📊 Intelligent data synthesis
- 🎨 Beautiful gradient UI
- 💡 Better user experience
- 🔍 Enhanced source display

## 📝 Files Created

- `app/backend/gemini_service.py` - Gemini AI integration
- `app/backend/config.py` - API configuration
- `app/GEMINI_INTEGRATION.md` - Detailed Gemini guide
- Enhanced `Chat.jsx` - Better UI with gradients
- Updated `app.py` - Gemini integration

## 🔐 API Key Configuration

The Gemini API key is configured in `app/backend/config.py`:
```python
GEMINI_API_KEY = "AIzaSyC1yEnlXvWwXPapThk76MgN47ZLRwfvMLY"
```

## 📈 Response Quality

**Basic Mode:**
> "In Andhra Pradesh, district X, rice produced 450 tons."

**Gemini Enhanced:**
> "Based on the agricultural data from Andhra Pradesh, specifically the East Godavari district, rice production during the 2010 Kharif season was substantial. The crop was cultivated across 150 hectares, yielding 450 tons. This represents a productive farming season with approximately 3 tons per hectare, indicating favorable growing conditions and effective agricultural practices."

## 🐛 Troubleshooting

**Backend won't start:**
- Check Python version (3.8+)
- Install dependencies: `pip install -r requirements.txt`
- Ensure `vector_database.pkl` exists

**Frontend won't start:**
- Check Node version (16+)
- Install dependencies: `npm install`
- Check port 3000 availability

**Gemini not working:**
- Check internet connection
- Verify API key in `config.py`
- Check API quota: https://makersuite.google.com/app/apikey

**Responses are slow:**
- Normal! Gemini takes 1-3 seconds
- Check internet speed
- Verify API key validity

## 📚 Documentation

- `README.md` - This file
- `SETUP.md` - Detailed setup guide
- `GEMINI_INTEGRATION.md` - Gemini AI documentation
- `TESTING_GUIDE.md` - Testing instructions
- `QUICK_START.md` - Quick start guide

## 🎉 Enjoy!

Start the app and try asking questions about Indian agriculture!

```bash
# Quick start
cd app/backend && python app.py
cd app/frontend && npm run dev

# Open browser
http://localhost:3000
```

---

**Status**: ✅ Fully Enhanced with Gemini AI
**Version**: 2.0
**Last Updated**: 2025
