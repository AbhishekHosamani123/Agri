# 🚀 How to Start the Application

## Dependencies are Already Installed! ✅

Now you just need to start both servers.

## Method 1: Using Separate Batch Files (Recommended)

### Step 1: Start the Backend
Double-click: **`start_backend.bat`**

Wait until you see:
```
Q&A System ready!
Running on http://0.0.0.0:5000
```

### Step 2: Start the Frontend  
Double-click: **`start_frontend.bat`**

Wait until you see:
```
Local: http://localhost:3000
```

### Step 3: Open Browser
Go to: **http://localhost:3000**

---

## Method 2: Manual Start

### Terminal 1 - Backend:
```bash
cd app/backend
python app.py
```

### Terminal 2 - Frontend:
```bash
cd app/frontend
npm run dev
```

---

## Troubleshooting

**"Port already in use" error:**
- Close any other applications using port 5000 or 3000

**Backend not starting:**
- Make sure Python is installed
- Check that `vector_database.pkl` is in the `app/backend/` folder

**Frontend not connecting:**
- Ensure backend is running first
- Look for error messages in the backend terminal

---

## Example Questions to Try

1. "What is the rice production in Andhra Pradesh?"
2. "Tell me about cotton production in Maharashtra"
3. "What crops are grown in Punjab?"
4. "What is the soil pH in Kerala?"

Enjoy! 🎉

