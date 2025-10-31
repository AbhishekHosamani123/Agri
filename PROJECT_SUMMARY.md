# Agriculture Q&A Web Application - Project Summary

## Overview

A complete web application that integrates the trained model from the `Model` folder with a modern chat interface from the `ChatBot` folder. The app provides an intelligent Q&A system for Indian agriculture data.

## What Was Created

### Project Structure

```
app/
├── README.md              # Main documentation
├── SETUP.md               # Quick setup guide
├── start.bat              # Windows startup script
├── .gitignore             # Git ignore rules
│
├── backend/               # Python Flask API Server
│   ├── app.py            # Main Flask application
│   ├── qa_system.py      # Q&A system logic
│   ├── requirements.txt  # Python dependencies
│   └── vector_database.pkl  # Trained model (copied from Model/)
│
└── frontend/             # React + Vite Frontend
    ├── index.html
    ├── package.json
    ├── vite.config.js
    ├── tailwind.config.js
    ├── postcss.config.js
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        └── components/
            └── Chat.jsx  # Main chat component
```

## Key Features

### Backend (Python Flask)
- ✅ REST API server with `/query` endpoint
- ✅ Intelligent Q&A system using vector embeddings
- ✅ Semantic search across crop and soil datasets
- ✅ Source tracking and confidence scoring
- ✅ Automatic model loading

### Frontend (React + Tailwind)
- ✅ Modern ChatGPT-style interface
- ✅ Real-time API integration
- ✅ Loading states and error handling
- ✅ Source display for each answer
- ✅ Responsive sidebar with conversation history
- ✅ Beautiful dark theme UI

## Integration Points

1. **Model Integration**: 
   - Copied `vector_database.pkl` from `Model/` folder
   - Adapted `qa_system.py` to work in app structure
   - Backend loads and uses the trained embeddings

2. **Chat UI Integration**:
   - Adapted ChatBot frontend to connect to real API
   - Replaced mock responses with actual API calls
   - Added source tracking and confidence display

## How It Works

1. **User asks a question** in the chat interface
2. **Frontend sends POST request** to `http://localhost:5000/query`
3. **Backend processes query** using vector similarity search
4. **QA system finds relevant chunks** from crop and soil data
5. **Answer is generated** with sources and confidence score
6. **Frontend displays response** in the chat interface

## Quick Start

1. **Install dependencies:**
   ```bash
   cd app/backend
   pip install -r requirements.txt
   
   cd ../frontend
   npm install
   ```

2. **Run the application:**
   
   Option A: Use the startup script
   ```bash
   # Windows
   start.bat
   ```
   
   Option B: Run manually
   ```bash
   # Terminal 1 - Backend
   cd app/backend
   python app.py
   
   # Terminal 2 - Frontend
   cd app/frontend
   npm run dev
   ```

3. **Open your browser:** `http://localhost:3000`

## Example Questions

- "What is the rice production in Andhra Pradesh?"
- "Tell me about cotton production in Maharashtra"
- "What crops are grown in Punjab?"
- "What is the soil pH in Kerala?"
- "Show soil health data for Rajasthan"
- "How much wheat was produced in Haryana?"

## Technology Stack

- **Backend**: Python, Flask, scikit-learn, pandas
- **Frontend**: React, Vite, TailwindCSS, Axios
- **Vector Search**: TF-IDF with cosine similarity
- **APIs**: RESTful API with JSON responses

## Files Modified/Created

### Created:
- `app/backend/app.py` - Flask API server
- `app/backend/qa_system.py` - Q&A system with path resolution
- `app/frontend/src/components/Chat.jsx` - Integrated chat component
- All config files (package.json, vite.config.js, etc.)

### Copied:
- `vector_database.pkl` from Model/ to app/backend/

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /stats` - System statistics  
- `POST /query` - Main query endpoint

## Troubleshooting

### Backend won't start
- Check Python version: `python --version` (need 3.8+)
- Install requirements: `pip install -r requirements.txt`
- Ensure `vector_database.pkl` is in `app/backend/`

### Frontend won't start
- Check Node version: `node --version` (need 16+)
- Install dependencies: `npm install`
- Check for port conflicts

### API connection errors
- Ensure backend is running on port 5000
- Check CORS settings in backend
- Verify API_URL in Chat.jsx

## Next Steps

To extend the application:

1. **Add new data sources** - Update `02_build_vector_database.py` in Model/
2. **Enhance UI** - Modify components in `frontend/src/`
3. **Add features** - Extend API in `backend/app.py`
4. **Deploy** - Build frontend with `npm run build`

## Files Location Reference

- Original model: `Model/03_qa_system.py`, `Model/04_api_server.py`
- Original chat UI: `ChatBot/src/components/Chat.jsx`
- Integrated app: `app/`
- Backend: `app/backend/`
- Frontend: `app/frontend/`

---

**Status**: ✅ Complete and Ready to Run

