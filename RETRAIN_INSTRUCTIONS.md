# 🚀 How to Retrain Your Model

## Step-by-Step Instructions

### 1. Open Your Terminal

Open PowerShell or Command Prompt in your **project root**:
```
E:\Data science Project For Gov\SaarthiAI
```

### 2. Navigate to Backend Folder

```bash
cd backend
```

### 3. Run the Retraining Script

```bash
python retrain_model.py
```

## What the Script Does

✨ **Advanced Training Process:**

1. **Loads Your Data**
   - Reads `Data Set/crop_production_full.csv` (~246K+ records)
   - Reads `Data Set/soil_health_complete_dataset.csv` (~9.7K+ records)

2. **Creates Smart Chunks**
   - Transforms raw data into informative text chunks
   - Each chunk includes: state, district, year, season, crop, area, production
   - Metadata includes all numeric values for statistical queries

3. **Advanced TF-IDF Vectorization**
   - Uses 1-3 word n-grams for better matching
   - Optimized for statistical keywords
   - 10,000+ vocabulary features
   - Logarithmic TF scaling for better performance

4. **Saves Optimized Model**
   - Creates `vector_database.pkl` with:
     - All chunks (crop + soil data)
     - Full metadata for aggregation
     - Trained TF-IDF vectorizer
     - Version 2.0 optimized

## Expected Output

You should see:
```
================================================================================
🌱 Loading Agricultural Datasets
================================================================================

📊 Loading crop production data from: Data Set/crop_production_full.csv
   Total records: 246,093
   Columns: ['state_name', 'district_name', 'crop_year', 'season', 'crop', 'area_', 'production_']

🌿 Loading soil health data from: Data Set/soil_health_complete_dataset.csv
   Total records: 9,785
   Columns: ['state_name', 'district_name', ..., 'test_year', 'sampling_season']

📦 Creating crop production chunks...
   Processed 10,000 crop records...
   Processed 20,000 crop records...
   ...
✅ Created 246,093 crop production chunks

🌿 Creating soil health chunks...
   Processed 5,000 soil records...
   ...
✅ Created 9,785 soil health chunks

📊 Total chunks created: 255,878
   - Crop production: 246,093
   - Soil health: 9,785

🔧 Creating optimized TF-IDF vectorizer...
   Fitting vectorizer on all chunks...
   Vocabulary size: 10,000
   Embedding shape: (255878, 10000)

💾 Saving vector database...
✅ Saved vector database to: vector_database.pkl
   File size: ~500 MB
   Chunks: 255,878
   Features: 10,000

================================================================================
✨ Model Training Successful!
================================================================================
```

## After Training

### 4. Restart Your Backend

```bash
python app.py
```

### 5. Test the Model

Try your questions again:
- "What was the total production of Rice in Bihar during the Kharif season?"
- "Which crop had the largest area in Maharashtra?"
- "What is the average yield of Sugarcane in Karnataka in 2010?"

## What Makes This Better?

### Previous Model Issues:
❌ Only 5-10 search results  
❌ Basic TF-IDF (1-gram only)  
❌ Limited metadata  
❌ Poor aggregation support  
❌ Small vocabulary  

### New Optimized Model:
✅ **255,878 chunks** of data  
✅ **Advanced TF-IDF** with 1-3 n-grams  
✅ **Rich metadata** for statistics  
✅ **Smart aggregation** for totals/averages/max  
✅ **10,000+ feature vocabulary**  
✅ **Optimized preprocessing**  

## Troubleshooting

### If you get "No module named pandas":
```bash
pip install pandas numpy scikit-learn
```

### If training is slow:
- This is normal! Processing 250K+ records takes time
- Expect 5-10 minutes for full training
- The process is verbose, so you can see progress

### If you get memory errors:
- Close other applications
- The final database will be ~500MB
- Ensure you have at least 2GB free RAM

## Files Created

After running, you'll have:
- ✅ `backend/vector_database.pkl` - Your new optimized model
- ✅ Updated with full dataset coverage
- ✅ Ready for complex statistical queries

## Verification

Run this to check your new model:
```bash
python check_database.py
```

You should see:
- ✅ Total chunks: ~255,878
- ✅ Bihar + Rice records: Many (if data exists)
- ✅ Kharif records: Many
- ✅ Multiple states and crops

---

**Time to Train**: ~5-10 minutes  
**Model Size**: ~500 MB  
**Status**: Ready for production use  

🎯 **Start training now!**



