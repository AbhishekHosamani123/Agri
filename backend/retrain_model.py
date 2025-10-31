"""
Advanced Model Retraining Script
Optimized for complex statistical queries with high performance
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class AdvancedModelTrainer:
    def __init__(self):
        self.crop_data = []
        self.soil_data = []
        self.chunks = []
        self.metadata = []
        self.vectorizer = None
        self.embeddings = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the datasets"""
        print("\n" + "="*80)
        print("🌱 Loading Agricultural Datasets")
        print("="*80)
        
        # Load crop production data
        crop_path = os.path.join('Data Set', 'crop_production_full.csv')
        print(f"\n📊 Loading crop production data from: {crop_path}")
        
        df_crop = pd.read_csv(crop_path)
        print(f"   Total records: {len(df_crop):,}")
        print(f"   Columns: {list(df_crop.columns)}")
        
        # Load soil health data
        soil_path = os.path.join('Data Set', 'soil_health_complete_dataset.csv')
        print(f"\n🌿 Loading soil health data from: {soil_path}")
        
        df_soil = pd.read_csv(soil_path)
        print(f"   Total records: {len(df_soil):,}")
        print(f"   Columns: {list(df_soil.columns)}")
        
        return df_crop, df_soil
    
    def create_crop_chunks(self, df):
        """Create informative chunks for crop production data"""
        print("\n📦 Creating crop production chunks...")
        
        chunks = []
        metadata_list = []
        
        # Group by state for better context
        for idx, row in df.iterrows():
            state = str(row['state_name']).strip()
            district = str(row['district_name']).strip()
            year = row['crop_year']
            season = str(row['season']).strip()
            crop = str(row['crop']).strip()
            area = row['area_']
            production = row['production_']
            
            # Skip if critical data is missing
            if pd.isna(area) or pd.isna(production) or pd.isna(year):
                continue
            
            # Create informative chunk text
            chunk_text = (
                f"In {state}, district {district}, during the {season} season of {int(year)}, "
                f"{crop} was cultivated on {area:.2f} hectares with a total production of {production:.2f} tons. "
                f"This data represents agricultural production in {state}, specifically in the {district} district."
            )
            
            # Create detailed metadata
            metadata = {
                'source': 'crop_production',
                'state': state,
                'district': district,
                'year': int(year),
                'season': season,
                'crop': crop,
                'area': float(area),
                'production': float(production),
                'record_id': idx
            }
            
            chunks.append(chunk_text)
            metadata_list.append(metadata)
            
            if len(chunks) % 10000 == 0:
                print(f"   Processed {len(chunks):,} crop records...")
        
        print(f"✅ Created {len(chunks):,} crop production chunks")
        return chunks, metadata_list
    
    def create_soil_chunks(self, df):
        """Create informative chunks for soil health data"""
        print("\n🌿 Creating soil health chunks...")
        
        chunks = []
        metadata_list = []
        
        for idx, row in df.iterrows():
            state = str(row['state_name']).strip()
            district = str(row['district_name']).strip()
            subdistrict = str(row['subdistrict_name']).strip()
            soil_type = str(row['soil_type']).strip()
            
            # Extract numeric values with validation
            ph_value = row['pH_value'] if pd.notna(row['pH_value']) else None
            organic_carbon = row['organic_carbon'] if pd.notna(row['organic_carbon']) else None
            nitrogen = row['nitrogen'] if pd.notna(row['nitrogen']) else None
            phosphorus = row['phosphorus'] if pd.notna(row['phosphorus']) else None
            potassium = row['potassium'] if pd.notna(row['potassium']) else None
            
            # Skip if critical data is missing
            if pd.isna(ph_value):
                continue
            
            # Create informative chunk text
            soil_info_parts = []
            if organic_carbon: soil_info_parts.append(f"organic carbon {organic_carbon:.2f}%")
            if nitrogen: soil_info_parts.append(f"nitrogen {nitrogen:.2f} kg/ha")
            if phosphorus: soil_info_parts.append(f"phosphorus {phosphorus:.2f} kg/ha")
            if potassium: soil_info_parts.append(f"potassium {potassium:.2f} kg/ha")
            
            nutrient_text = ", ".join(soil_info_parts) if soil_info_parts else "standard nutrient levels"
            
            chunk_text = (
                f"In {state}, district {district} (subdistrict: {subdistrict}), "
                f"the soil type is {soil_type} with a pH value of {ph_value:.2f}. "
                f"The soil contains: {nutrient_text}. "
                f"This data represents soil health information for {state}, {district}."
            )
            
            # Create detailed metadata
            metadata = {
                'source': 'soil_health',
                'state': state,
                'district': district,
                'subdistrict': subdistrict,
                'soil_type': soil_type,
                'pH': float(ph_value) if ph_value else None,
                'organic_carbon': float(organic_carbon) if organic_carbon else None,
                'nitrogen': float(nitrogen) if nitrogen else None,
                'phosphorus': float(phosphorus) if phosphorus else None,
                'potassium': float(potassium) if potassium else None,
                'record_id': idx
            }
            
            chunks.append(chunk_text)
            metadata_list.append(metadata)
            
            if len(chunks) % 5000 == 0:
                print(f"   Processed {len(chunks):,} soil records...")
        
        print(f"✅ Created {len(chunks):,} soil health chunks")
        return chunks, metadata_list
    
    def create_vectorizer(self, chunks):
        """Create optimized TF-IDF vectorizer"""
        print("\n🔧 Creating optimized TF-IDF vectorizer...")
        
        # Advanced TF-IDF with better features for statistical queries
        vectorizer = TfidfVectorizer(
            max_features=10000,  # More features for better discrimination
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            lowercase=True,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\b\w+\b',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True  # Logarithmic TF for better performance
        )
        
        print("   Fitting vectorizer on all chunks...")
        embeddings = vectorizer.fit_transform(chunks)
        
        print(f"   Vocabulary size: {len(vectorizer.vocabulary_):,}")
        print(f"   Embedding shape: {embeddings.shape}")
        
        return vectorizer, embeddings
    
    def train_model(self):
        """Main training function"""
        print("\n" + "="*80)
        print("🚀 Starting Advanced Model Training")
        print("="*80)
        
        # Load data
        df_crop, df_soil = self.load_and_preprocess_data()
        
        # Create chunks
        crop_chunks, crop_metadata = self.create_crop_chunks(df_crop)
        soil_chunks, soil_metadata = self.create_soil_chunks(df_soil)
        
        # Combine all chunks
        all_chunks = crop_chunks + soil_chunks
        all_metadata = crop_metadata + soil_metadata
        
        print(f"\n📊 Total chunks created: {len(all_chunks):,}")
        print(f"   - Crop production: {len(crop_chunks):,}")
        print(f"   - Soil health: {len(soil_chunks):,}")
        
        # Create vectorizer and embeddings
        self.vectorizer, self.embeddings = self.create_vectorizer(all_chunks)
        
        # Save to vector database
        self.save_vector_database(all_chunks, all_metadata)
        
        print("\n" + "="*80)
        print("✅ Training Complete!")
        print("="*80)
        
    def save_vector_database(self, chunks, metadata):
        """Save the trained model to disk"""
        print("\n💾 Saving vector database...")
        
        vector_db = {
            'chunks': chunks,
            'metadata': metadata,
            'embeddings': self.embeddings,
            'vectorizer': self.vectorizer,
            'method': 'tf-idf_advanced',
            'n_features': self.embeddings.shape[1],
            'n_chunks': len(chunks),
            'trained_date': datetime.now().isoformat(),
            'version': '2.0_optimized'
        }
        
        output_path = 'vector_database.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(vector_db, f)
        
        print(f"✅ Saved vector database to: {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        print(f"   Chunks: {len(chunks):,}")
        print(f"   Features: {self.embeddings.shape[1]:,}")

def main():
    print("\n" + "="*80)
    print("🌾 Advanced Agricultural AI Model Trainer")
    print("="*80)
    print("Optimized for complex statistical queries with high performance")
    print("="*80)
    
    trainer = AdvancedModelTrainer()
    
    try:
        trainer.train_model()
        
        print("\n" + "="*80)
        print("✨ Model Training Successful!")
        print("="*80)
        print("\nThe model is now ready to use. Start your backend with:")
        print("  python app.py")
        print("\nThe improved model can now answer complex questions like:")
        print("  - Total production calculations")
        print("  - Largest/smallest comparisons")
        print("  - Average yield calculations")
        print("  - Multi-year aggregations")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

