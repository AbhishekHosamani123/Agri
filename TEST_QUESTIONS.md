# Sample Test Questions for Agriculture Q&A Model

Use these questions to test the system and compare with expected answers.

## 🌾 Crop Production Questions

### Question 1:
**Ask:** "What is the rice production in Andhra Pradesh?"

**Expected Answer Type:**
- Should mention specific district
- Production amount in tons
- Area in hectares
- Year of production
- Example: "In Andhra Pradesh, district [district name], rice was produced in [year] with production of [X] tons from [Y] hectares."

---

### Question 2:
**Ask:** "Tell me about cotton production in Maharashtra"

**Expected Answer Type:**
- Multiple records mentioning different districts
- Production statistics
- May mention multiple years
- Example: "Found multiple crop production records. Top crops: cotton. Average production: [X] tons across [Y] states."

---

### Question 3:
**Ask:** "What crops are grown in Punjab?"

**Expected Answer Type:**
- Lists various crops grown in Punjab
- Production amounts
- May cover multiple years and districts
- Example: "Found multiple crop production records. Top crops: wheat, rice, cotton. Average production: [X] tons across [Y] districts."

---

### Question 4:
**Ask:** "Show me rice production data"

**Expected Answer Type:**
- Rice production statistics
- Multiple states/districts
- Production and area information
- Example: Rice production records with detailed statistics

---

### Question 5:
**Ask:** "How much wheat was produced in Haryana in 2010?"

**Expected Answer Type:**
- Specific year (2010)
- Wheat production in Haryana
- Exact tonnage and area
- Example: "In Haryana, district [name], wheat was produced in 2010 with production of [X] tons from [Y] hectares."

---

## 🌱 Soil Health Questions

### Question 6:
**Ask:** "What is the soil pH in Kerala?"

**Expected Answer Type:**
- pH level mentioned
- Specific districts in Kerala
- May include other soil properties
- Example: "In Kerala, district [name], the soil is [type] type with pH [X]..."

---

### Question 7:
**Ask:** "Show soil health data for Rajasthan"

**Expected Answer Type:**
- Soil properties (pH, nutrients)
- Districts in Rajasthan
- Organic carbon, nitrogen, phosphorus, potassium levels
- Example: "Found soil health data across [X] districts. Average pH: [Y], Average Organic Carbon: [Z]%."

---

### Question 8:
**Ask:** "Tell me about organic carbon levels in Indian soil"

**Expected Answer Type:**
- Organic carbon percentages
- Multiple states/districts
- Average values
- Example: "Found soil health data with average organic carbon: [X]%"

---

### Question 9:
**Ask:** "What is the nitrogen content in soil?"

**Expected Answer Type:**
- Nitrogen levels in kg/ha
- Various locations
- Example: "In [state], district [name], ... Nitrogen: [X] kg/ha"

---

### Question 10:
**Ask:** "Compare soil pH across different states"

**Expected Answer Type:**
- Multiple states mentioned
- pH values for different states
- Comparative statistics
- Example: "Found soil health data across [states]. Average pH: [X]"

---

## 🔄 Combination Questions

### Question 11:
**Ask:** "What is the rice production and soil health in Andhra Pradesh?"

**Expected Answer Type:**
- Combines both crop and soil data
- Production information
- Soil properties
- Example: Shows both crop production and soil health data for Andhra Pradesh

---

### Question 12:
**Ask:** "Tell me about agriculture in Maharashtra"

**Expected Answer Type:**
- Crop production data
- Soil health information
- Multiple data points
- Example: Comprehensive agricultural information about Maharashtra

---

## ❌ Questions That May Not Work

### Question 13:
**Ask:** "What is the weather like?"

**Expected Response:**
- "I couldn't find relevant information to answer your question in the available datasets."
- Reason: Weather data not included

---

### Question 14:
**Ask:** "Who is the agriculture minister?"

**Expected Response:**
- "I couldn't find relevant information..."
- Reason: Political/current events data not included

---

## 📊 Understanding the Output

### Key Components of Responses:

1. **Answer Section:**
   - Main answer with statistics
   - Natural language summary
   - Specific numbers (tons, hectares, percentages)

2. **Sources Section:**
   - Dataset source (crop_production or soil_health)
   - Relevance score (confidence percentage)
   - Specific details like state, district, crop, year

3. **Confidence Score:**
   - Higher score = better match
   - Usually between 0.2 to 0.8 for good matches
   - Below 0.1 = no relevant data found

---

## 🧪 Testing Checklist

- [ ] Try Questions 1-3 (crop production)
- [ ] Try Questions 6-8 (soil health)
- [ ] Try Question 11 (combination)
- [ ] Verify sources are displayed
- [ ] Check confidence scores
- [ ] Test with Question 13 (should not work)

---

## 💡 Tips for Best Results

1. **Be specific**: "rice production in Andhra Pradesh" works better than just "rice"
2. **Use state/district names**: The model has location-based data
3. **Ask about crops**: wheat, rice, cotton, sugarcane, etc.
4. **Ask about soil properties**: pH, nitrogen, phosphorus, potassium
5. **Combine queries**: You can ask for both crop and soil data together

---

## Expected Response Format

```json
{
  "question": "What is the rice production in Andhra Pradesh?",
  "answer": "In Andhra Pradesh, district [X], rice was produced in [Y] with production of [Z] tons...",
  "confidence": 0.XX,
  "sources": [
    {
      "dataset": "crop_production",
      "relevance": "XX%",
      "details": { state, district, crop, year, etc. }
    }
  ],
  "num_results": X
}
```

---

## 🎯 Quick Test Sequence

1. Start with Question 1 (rice in Andhra Pradesh)
2. Try Question 6 (soil pH in Kerala)
3. Try Question 11 (combination question)
4. Check that sources are displayed
5. Verify confidence scores are reasonable (>0.2 for good matches)

Good luck testing! 🚀

