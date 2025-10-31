# Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Install Backend Dependencies
```bash
cd app/backend
pip install -r requirements.txt
```

### Step 2: Install Frontend Dependencies
```bash
cd app/frontend
npm install
```

### Step 3: Run the Application

**Windows:** Double-click `start.bat` or run:
```bash
cd app
start.bat
```

**Manual Method:**
```bash
# Terminal 1 - Start Backend
cd app/backend
python app.py

# Terminal 2 - Start Frontend  
cd app/frontend
npm run dev
```

Then open: **http://localhost:3000**

---

## 📂 What Was Integrated

```
Original Files → New App Structure

Model/
├── 03_qa_system.py       →  app/backend/qa_system.py
├── 04_api_server.py      →  app/backend/app.py
└── vector_database.pkl   →  app/backend/vector_database.pkl

ChatBot/
├── src/components/
│   └── Chat.jsx          →  app/frontend/src/components/Chat.jsx
└── (all other files)     →  app/frontend/*

Result: Complete Web App!
```

---

## 🎯 Architecture

```
┌─────────────────┐
│   Browser       │
│  (Port 3000)    │
└────────┬────────┘
         │ HTTP Requests
         ▼
┌─────────────────┐
│  React Frontend │
│  (Chat UI)      │
└────────┬────────┘
         │ API Calls
         ▼
┌─────────────────┐
│  Flask Backend  │
│  (Port 5000)    │
└────────┬────────┘
         │ Queries
         ▼
┌─────────────────┐
│  Q&A System     │
│  (Vector DB)    │
└────────┬────────┘
         │ Searches
         ▼
┌─────────────────┐
│  Crop & Soil    │
│  Data Files     │
└─────────────────┘
```

---

## 💡 Example Usage

1. Open http://localhost:3000
2. Ask: "What is rice production in Andhra Pradesh?"
3. Get instant answer with sources!

---

## 📝 Key Files

| File | Purpose |
|------|---------|
| `app/backend/app.py` | Flask API server |
| `app/backend/qa_system.py` | Q&A logic |
| `app/frontend/src/components/Chat.jsx` | Chat UI |
| `app/start.bat` | Easy startup |

---

## ⚠️ Troubleshooting

**Problem:** Can't connect to backend  
**Solution:** Ensure Python backend is running on port 5000

**Problem:** Module not found  
**Solution:** Run `pip install -r requirements.txt` in backend folder

**Problem:** npm errors  
**Solution:** Run `npm install` in frontend folder

---

Ready to use! 🎉

