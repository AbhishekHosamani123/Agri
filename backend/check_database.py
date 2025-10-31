"""
Script to check what data is in the vector database
"""

import pickle
import os

def check_database():
    """Check what's in the vector database"""
    
    # Find the database file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, 'vector_database.pkl')
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return
    
    print(f"✅ Loading database from: {db_path}")
    
    with open(db_path, 'rb') as f:
        vector_db = pickle.load(f)
    
    chunks = vector_db['chunks']
    metadata = vector_db['metadata']
    
    print(f"\n📊 Total chunks in database: {len(chunks)}")
    
    # Check for Bihar and Rice
    bihar_count = 0
    rice_count = 0
    bihar_rice_count = 0
    kharif_count = 0
    bihar_kharif_count = 0
    
    crops_found = set()
    states_found = set()
    seasons_found = set()
    
    print("\n🔍 Scanning database...")
    
    for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
        source = meta.get('source', 'unknown')
        
        # Check for Bihar
        if 'bihar' in str(meta).lower() or 'bihar' in chunk.lower():
            bihar_count += 1
            
            # Check for Kharif in Bihar
            if 'kharif' in str(meta).lower() or 'kharif' in chunk.lower():
                bihar_kharif_count += 1
                if i < 5:  # Show first 5 Bihar Kharif results
                    print(f"\n  Bihar Kharif #{bihar_kharif_count}:")
                    print(f"    Metadata: {meta}")
                    print(f"    Chunk preview: {chunk[:100]}...")
        
        # Check for Rice
        if 'rice' in chunk.lower():
            rice_count += 1
            
            # Check for Bihar + Rice
            if 'bihar' in str(meta).lower() or 'bihar' in chunk.lower():
                bihar_rice_count += 1
                if bihar_rice_count <= 5:  # Show first 5
                    print(f"\n  Bihar + Rice #{bihar_rice_count}:")
                    print(f"    Chunk: {chunk[:200]}")
        
        # Check for Kharif
        if 'kharif' in str(meta).lower() or 'kharif' in chunk.lower():
            kharif_count += 1
        
        # Collect crops, states, seasons
        if 'crop' in meta:
            crops_found.add(meta['crop'].lower())
        if 'state' in meta:
            states_found.add(meta['state'].lower())
        if 'season' in meta:
            seasons_found.add(meta['season'].lower())
    
    # Print statistics
    print(f"\n{'='*60}")
    print("DATABASE STATISTICS")
    print(f"{'='*60}")
    print(f"\n📍 Bihar records: {bihar_count}")
    print(f"🌾 Rice records: {rice_count}")
    print(f"🌾 Bihar + Rice records: {bihar_rice_count}")
    print(f"🌱 Kharif records: {kharif_count}")
    print(f"📍 Bihar + Kharif records: {bihar_kharif_count}")
    
    print(f"\n📊 Unique crops found: {len(crops_found)}")
    print(f"Sample crops: {', '.join(list(crops_found)[:10])}")
    
    print(f"\n🗺️  Unique states found: {len(states_found)}")
    print(f"Sample states: {', '.join(list(states_found)[:15])}")
    
    print(f"\n🌍 Seasons found: {', '.join(seasons_found) if seasons_found else 'None'}")
    
    # Check sample metadata structure
    print(f"\n📋 Sample metadata structure:")
    if metadata:
        sample = metadata[0]
        print(f"\n  Keys: {sample.keys()}")
        print(f"  Example: {sample}")
    
    # Search for specific terms
    print(f"\n{'='*60}")
    print("SEARCHING FOR SPECIFIC DATA")
    print(f"{'='*60}")
    
    search_terms = ['bihar', 'rice', 'kharif', 'paddy']
    
    for term in search_terms:
        count = 0
        for chunk in chunks:
            if term in chunk.lower():
                count += 1
        print(f"'{term}': found in {count} chunks")
    
    # Show some random samples
    print(f"\n{'='*60}")
    print("RANDOM CHUNK SAMPLES")
    print(f"{'='*60}")
    
    import random
    sample_indices = random.sample(range(min(10, len(chunks))), min(3, len(chunks)))
    
    for idx in sample_indices:
        print(f"\nChunk #{idx}:")
        print(f"  Text: {chunks[idx][:150]}...")
        if metadata and idx < len(metadata):
            print(f"  Metadata: {metadata[idx]}")
    
    print(f"\n{'='*60}")
    
    # Diagnosis
    print("\n💡 DIAGNOSIS:")
    
    if bihar_rice_count == 0:
        print("❌ NO Rice data found for Bihar!")
        print("   → This explains why queries about Rice in Bihar fail.")
    
    if bihar_kharif_count < 10:
        print("⚠️  Very few Bihar Kharif records found!")
        print(f"   → Only {bihar_kharif_count} records.")
    
    if rice_count == 0:
        print("❌ NO Rice data found in database!")
    
    if len(crops_found) < 5:
        print("⚠️  Very limited crop variety in database!")
    
    print("\n📝 RECOMMENDATION:")
    
    if bihar_rice_count == 0 or len(crops_found) < 20:
        print("→ The vector database appears to have limited or incomplete data.")
        print("→ You need to rebuild the vector database with complete agricultural datasets.")
        print("→ Contact your data source provider or check your CSV files.")
    else:
        print("→ Data exists but search relevance may be low.")
        print("→ Try rephrasing questions or checking data structure.")

if __name__ == '__main__':
    check_database()




