# Comprehensive Testing Guide

## 🎯 Test Questions with Expected Answers

Based on the model's training data, here are specific questions you can ask and what to expect in the responses.

---

## ✅ Tier 1: Basic Crop Production Questions

### Test 1: Rice in Andhra Pradesh
**Question:** `"What is the rice production in Andhra Pradesh?"`

**Expected Response Format:**
```
✅ Should contain:
- State: "Andhra Pradesh" or "Andhra"
- Crop: "Rice" or "rice"
- Specific district names
- Production amounts (in tons)
- Area (in hectares)
- Year and season information

Example chunks you'll see:
"In Andhra Pradesh, district [X], during [year] [season] season, rice was cultivated on [X] hectares with a production of [Y] tons."

Confidence Score: 0.3 - 0.6
Sources: crop_production dataset
```

---

### Test 2: Cotton in Maharashtra
**Question:** `"Tell me about cotton production in Maharashtra"`

**Expected Response Format:**
```
✅ Should contain:
- Multiple districts in Maharashtra
- Cotton production statistics
- Various years and seasons
- Specific production amounts

Example: "Found multiple crop production records. Top crops: cotton. Average production: [X] tons across [Y] districts."

Confidence Score: 0.3 - 0.7
```

---

### Test 3: Crops in Punjab
**Question:** `"What crops are grown in Punjab?"`

**Expected Response Format:**
```
✅ Should contain:
- Multiple crop types (likely wheat, rice, cotton, sugarcane)
- Production statistics
- Different districts
- Multiple years

Example: Multiple crop records showing wheat, rice, cotton, etc.

Confidence Score: 0.4 - 0.7
```

---

### Test 4: Wheat in Haryana
**Question:** `"What is the wheat production in Haryana?"`

**Expected Response Format:**
```
✅ Should contain:
- Haryana districts
- Wheat production data
- Area and production figures

Example: Specific wheat production records from Haryana districts

Confidence Score: 0.3 - 0.6
```

---

## ✅ Tier 2: Soil Health Questions

### Test 5: Soil pH in Kerala
**Question:** `"What is the soil pH in Kerala?"`

**Expected Response Format:**
```
✅ Should contain:
- Districts in Kerala
- pH values (typically 5.5 - 7.0 for Kerala)
- pH status (acidic/neutral/alkaline)
- Other soil parameters

Example: "In Kerala, district [X], village [Y], the soil type is [type] with pH value of [X] ([status])."

Confidence Score: 0.3 - 0.6
Sources: soil_health dataset
```

---

### Test 6: Soil Data for Rajasthan
**Question:** `"Show soil health data for Rajasthan"`

**Expected Response Format:**
```
✅ Should contain:
- Rajasthan districts
- Soil parameters (pH, organic carbon, nutrients)
- Village names
- Test year information

Example: "Found soil health data across [X] districts. Average pH: [Y], Average Organic Carbon: [Z]%."

Confidence Score: 0.4 - 0.7
```

---

### Test 7: Organic Carbon Levels
**Question:** `"What is the organic carbon content in Indian soil?"`

**Expected Response Format:**
```
✅ Should contain:
- Organic carbon percentages (typically 0.3 - 1.5%)
- Multiple states/districts
- Average values
- Specific values for different locations

Confidence Score: 0.2 - 0.5
```

---

### Test 8: Nitrogen Content
**Question:** `"Tell me about nitrogen levels in soil"`

**Expected Response Format:**
```
✅ Should contain:
- Nitrogen levels in kg/ha (typically 100-300 kg/ha)
- Multiple locations
- State/district information

Example: "In [state], district [X], ... Nitrogen: [Y] kg/ha"

Confidence Score: 0.2 - 0.5
```

---

## ✅ Tier 3: Combined Questions

### Test 9: Crop and Soil Together
**Question:** `"What is the rice production and soil health in Andhra Pradesh?"`

**Expected Response Format:**
```
✅ Should contain BOTH:
- Crop production data (rice in Andhra Pradesh)
- Soil health data (from Andhra Pradesh)

Example: Two-part answer showing both crop production AND soil health information.

Confidence Score: 0.3 - 0.6
Sources: BOTH crop_production AND soil_health
```

---

### Test 10: General Agriculture Query
**Question:** `"Tell me about agriculture in Tamil Nadu"`

**Expected Response Format:**
```
✅ Should contain:
- Crop production data from Tamil Nadu
- Soil health data from Tamil Nadu
- Multiple datasets combined

Example: Comprehensive agricultural information about Tamil Nadu

Confidence Score: 0.4 - 0.7
Sources: Both datasets
```

---

## ⚠️ Tier 4: Queries That SHOULDN'T Work

### Test 11: Weather Query
**Question:** `"What is the weather like?"`

**Expected Response:**
```
❌ Should return: "I couldn't find relevant information to answer your question in the available datasets."

Reason: Weather data not included in training
```

---

### Test 12: Future Predictions
**Question:** `"What will be the rice production next year?"`

**Expected Response:**
```
❌ Should return: No relevant information or mentions only historical data up to 2015

Reason: Data only covers years 1997-2015
```

---

## 📊 Response Quality Checklist

When testing, verify:

### ✅ GOOD Response Indicators:
- [ ] Answer mentions specific states/districts
- [ ] Contains actual numbers (tons, hectares, percentages)
- [ ] Includes year information
- [ ] Sources are listed
- [ ] Confidence score > 0.2
- [ ] Answer is relevant to the question

### ❌ POOR Response Indicators:
- [ ] Generic answer with no specifics
- [ ] Low confidence score (< 0.2)
- [ ] No sources listed
- [ ] Answer doesn't match the question

---

## 🎯 Confidence Score Guide

- **0.6 - 1.0**: Excellent match (very specific query, exact data found)
- **0.4 - 0.6**: Good match (relevant data retrieved)
- **0.2 - 0.4**: Acceptable match (somewhat relevant)
- **0.0 - 0.2**: Poor match (no relevant data or very generic)

---

## 💡 Pro Tips for Testing

1. **Be Specific**: 
   - ✅ "rice production in Andhra Pradesh"
   - ❌ "agriculture data"

2. **Use Correct State Names**:
   - Try: "Maharashtra", "Tamil Nadu", "Kerala"
   - Avoid abbreviations

3. **Ask About Known Crops**:
   - Rice, Wheat, Cotton, Sugarcane
   - Fruits: Banana, Orange
   - Pulses: Chickpea, Lentil

4. **Test Different Years**:
   - Data covers 1997-2015
   - Try: "crops in 2010" or "production in 2005"

5. **Combine Keywords**:
   - "rice production in Andhra Pradesh"
   - "soil pH in Kerala"
   - "wheat area in Punjab"

---

## 📋 Quick Testing Sequence

### Basic Test (Should Work):
1. Ask: "What is rice production in Andhra Pradesh?"
   - ✅ Expect: Specific production data
   - ✅ Confidence: > 0.3

2. Ask: "What is soil pH in Kerala?"
   - ✅ Expect: pH values and districts
   - ✅ Confidence: > 0.3

3. Ask: "Tell me about wheat in Haryana?"
   - ✅ Expect: Wheat production statistics
   - ✅ Confidence: > 0.3

### Advanced Test (Should Work):
4. Ask: "Show me agriculture data for Maharashtra"
   - ✅ Expect: Both crop and soil data
   - ✅ Confidence: > 0.4
   - ✅ Sources: Multiple sources

### Edge Cases (Should Not Work):
5. Ask: "What's the weather?"
   - ❌ Expect: "Couldn't find relevant information"

---

## 🔍 What the UI Will Show

For each response, you'll see:

```
┌─────────────────────────────────────┐
│ Question: "What is rice in AP?"      │
├─────────────────────────────────────┤
│ Answer: "In Andhra Pradesh, district│
│ East Godavari..."                    │
│                                      │
│ Sources:                             │
│ • crop_production - 45% relevance    │
│ • crop_production - 38% relevance    │
└─────────────────────────────────────┘
```

---

## 📈 Expected Performance

- **Query Time**: < 2 seconds
- **Response Length**: 1-5 sentences
- **Average Confidence**: 0.3 - 0.6
- **Source Count**: 1-5 sources per query

---

## ✅ Success Criteria

A successful test shows:
1. ✅ Relevant answer to the question
2. ✅ Specific numbers and statistics
3. ✅ Source citations displayed
4. ✅ Confidence score visible
5. ✅ Fast response (< 2 seconds)

---

**Happy Testing!** 🎉

Try these questions when you start the app and compare the answers!

