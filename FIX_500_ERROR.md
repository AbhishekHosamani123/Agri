# Fix for 500 Error

## ✅ What I Fixed

1. **Added better error handling** - The backend now properly handles errors
2. **Disabled Gemini by default** - So the app works without API issues
3. **Added logging** - You can see what's happening in the backend console
4. **Improved error messages** - Better debugging information

## 🚀 How to Restart

### Step 1: Stop Current Servers
Press `Ctrl+C` in both terminal windows to stop the servers

### Step 2: Restart Backend
```bash
cd app/backend
python app.py
```

You should see:
```
Starting Q&A API Server...
✅ Q&A System ready!
🚀 Server will start on: http://localhost:5000
```

### Step 3: Restart Frontend
```bash
cd app/frontend
npm run dev
```

### Step 4: Test
Open: http://localhost:3000

Ask: "What is rice production in Andhra Pradesh?"

## 🔍 If Still Having Issues

Check the backend terminal for error messages. Common issues:

1. **Module not found**
   ```bash
   pip install -r requirements.txt
   ```

2. **Vector database missing**
   - Make sure `vector_database.pkl` is in `app/backend/`

3. **Port already in use**
   - Close other applications using ports 5000/3000

## ✅ What Should Work Now

- ✅ Basic Q&A system (without Gemini)
- ✅ Fast responses
- ✅ Source citations
- ✅ Confidence scores
- ✅ No more 500 errors

## 🎯 Test Questions

1. "What is rice production in Andhra Pradesh?"
2. "What crops are grown in Punjab?"
3. "What is the soil pH in Kerala?"

All should work without errors!

