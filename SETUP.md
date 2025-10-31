# Quick Setup Guide

Follow these steps to get the Agriculture Q&A App running.

## Prerequisites

Before starting, make sure you have:
- Python 3.8 or higher installed
- Node.js 16 or higher installed
- npm (comes with Node.js)

## Installation Steps

### 1. Install Backend Dependencies

Open a terminal in the `app/backend` directory and run:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

Open a new terminal in the `app/frontend` directory and run:

```bash
cd frontend
npm install
```

## Running the Application

### Option 1: Using the Startup Script (Windows)

Simply double-click `start.bat` in the `app` folder, or run from terminal:

```bash
start.bat
```

This will start both servers automatically.

### Option 2: Manual Start

#### Terminal 1 - Backend:
```bash
cd app/backend
python app.py
```

#### Terminal 2 - Frontend:
```bash
cd app/frontend
npm run dev
```

## Access the Application

- Frontend: Open your browser and go to `http://localhost:3000`
- Backend API: Available at `http://localhost:5000`

## Troubleshooting

### "Module not found" error
- Make sure you installed all dependencies with `pip install -r requirements.txt`
- Try creating a virtual environment:
  ```bash
  python -m venv venv
  venv\Scripts\activate  # Windows
  pip install -r requirements.txt
  ```

### "Cannot connect to backend"
- Ensure the backend is running on port 5000
- Check the terminal for any error messages

### Port already in use
- Backend uses port 5000: If it's busy, change it in `app.py`
- Frontend uses port 3000: If it's busy, Vite will automatically use another port

### "Vector database not found"
- Ensure `vector_database.pkl` is in the `app/backend/` directory
- If missing, copy it from the `Model/` folder

## Testing the API

You can test the backend directly using curl:

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is rice production in Andhra Pradesh?\"}"
```

Or use a tool like Postman to test the `/query` endpoint.

## Example Questions

Try asking:
- "What is the rice production in Andhra Pradesh?"
- "Tell me about cotton production in Maharashtra"
- "What crops are grown in Punjab?"
- "What is the soil pH in Kerala?"
- "Show soil health data for Rajasthan"

Enjoy using the Agriculture Q&A Assistant!

