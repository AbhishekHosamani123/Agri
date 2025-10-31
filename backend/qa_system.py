"""
Intelligent Q&A System with Source Tracking
Retrieves relevant information from multiple data sources and combines them
"""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

class IntelligentQASystem:
    def __init__(self, vector_db_path='vector_database.pkl'):
        """Initialize the Q&A system"""
        # Get the directory where this file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find vector database in current directory first
        full_path = os.path.join(current_dir, vector_db_path)
        
        if not os.path.exists(full_path):
            # Try Model directory in parent
            parent_dir = os.path.dirname(os.path.dirname(current_dir))
            model_path = os.path.join(parent_dir, 'Model', 'vector_database.pkl')
            if os.path.exists(model_path):
                full_path = model_path
                print(f"Found vector database at: {full_path}")
            else:
                raise FileNotFoundError(f"Vector database not found at {full_path}")
        
        print(f"Loading vector database from {full_path}...")
        with open(full_path, 'rb') as f:
            self.vector_db = pickle.load(f)
        
        self.embeddings = self.vector_db['embeddings']
        self.chunks = self.vector_db['chunks']
        self.metadata = self.vector_db['metadata']
        self.vectorizer = self.vector_db['vectorizer']
        
        print(f"Loaded {len(self.chunks)} knowledge chunks")
        print(f"Method: {self.vector_db['method']}")
    
    def search(self, query, top_k=5):
        """Search for relevant chunks based on query"""
        # Vectorize the query
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarity scores
        similarities = cosine_similarity(query_vector, self.embeddings).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk': self.chunks[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarities[idx])
            })
        
        return results
    
    def answer_question(self, question, top_k=10):
        """Generate an answer with proper citations"""
        
        # Search for relevant chunks with more results
        search_results = self.search(question, top_k=top_k)
        
        # Even if similarity is low, pass the question to Gemini for general guidance
        # We'll let Gemini handle both specific data questions and general agriculture questions
        if not search_results or search_results[0]['similarity'] < 0.05:
            # Return minimal response - Gemini will enhance it with general guidance
            return {
                'answer': "",  # Empty - will be filled by Gemini for general questions
                'sources': [],
                'confidence': 0,
                'search_results_count': 0,
                'is_general_question': True
            }
        
        # Use all results with similarity > 0.1
        relevant_results = [r for r in search_results if r['similarity'] > 0.1]
        
        # Analyze the results and generate an answer
        answer_parts = []
        sources = []
        
        # Group by source type
        crop_data = []
        soil_data = []
        
        for result in relevant_results if relevant_results else search_results:
            source_info = {
                'text': result['chunk'],
                'metadata': result['metadata'],
                'similarity': result['similarity']
            }
            
            if result['metadata']['source'] == 'crop_production':
                crop_data.append(source_info)
            elif result['metadata']['source'] == 'soil_health':
                soil_data.append(source_info)
            
            sources.append({
                'dataset': result['metadata']['source'],
                'details': result['metadata'],
                'relevance': f"{result['similarity']:.2%}",
                'chunk': result['chunk']
            })
        
        # Generate answer based on findings
        if crop_data:
            answer_parts.append(self._format_crop_answer(crop_data, question))
        
        if soil_data:
            answer_parts.append(self._format_soil_answer(soil_data))
        
        if crop_data and soil_data:
            answer = f"Based on the available data:\n{chr(10).join(answer_parts)}\n\nThis information is derived from multiple data sources."
        elif answer_parts:
            answer = chr(10).join(answer_parts)
        else:
            answer = "I couldn't find specific information to answer your question in the available datasets."
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': float(search_results[0]['similarity']),
            'search_results_count': len(relevant_results if relevant_results else search_results)
        }
    
    def _format_crop_answer(self, crop_data, question=""):
        """Format crop production information"""
        if not crop_data:
            return ""
        
        if len(crop_data) == 1:
            # Single result
            main_result = crop_data[0]['metadata']
            return f"In {main_result['state']}, district {main_result['district']}, {main_result['crop']} was produced in {main_result['year']} with production of {main_result['production']:.2f} tons from {main_result['area']:.2f} hectares."
        
        # Multiple results - analyze based on question
        states = set([d['metadata']['state'] for d in crop_data])
        crops = set([d['metadata']['crop'] for d in crop_data])
        
        # Check if question asks for totals, averages, or largest
        question_lower = question.lower()
        
        if 'total' in question_lower or 'sum' in question_lower:
            # Sum all production values
            total_production = sum([d['metadata']['production'] for d in crop_data])
            return f"Total production across {len(crop_data)} records: {total_production:.2f} tons."
        
        if 'largest' in question_lower or 'highest' in question_lower or 'maximum' in question_lower:
            # Find the record with highest value
            max_result = max(crop_data, key=lambda x: x['metadata'].get('production', 0) or x['metadata'].get('area', 0))
            main_result = max_result['metadata']
            return f"Largest production found: {main_result['crop']} in {main_result['state']} district {main_result['district']} with {main_result['production']:.2f} tons from {main_result['area']:.2f} hectares."
        
        if 'average' in question_lower or 'avg' in question_lower:
            # Calculate average
            avg_production = np.mean([d['metadata']['production'] for d in crop_data])
            avg_area = np.mean([d['metadata']['area'] for d in crop_data])
            return f"Average production: {avg_production:.2f} tons from {avg_area:.2f} hectares across {len(states)} states."
        
        # Default summary
        return f"Found {len(crop_data)} crop production records across {len(states)} states. Crops: {', '.join(list(crops)[:5])}. Average production: {np.mean([d['metadata']['production'] for d in crop_data]):.2f} tons."
    
    def _format_soil_answer(self, soil_data):
        """Format soil health information"""
        if not soil_data:
            return ""
        
        if len(soil_data) == 1:
            # Single result
            main_result = soil_data[0]['metadata']
            return f"In {main_result['state']}, district {main_result['district']}, the soil is {main_result['soil_type']} type with pH {main_result['pH']:.2f}, Organic Carbon: {main_result['organic_carbon']:.2f}%, Nitrogen: {main_result['nitrogen']:.2f} kg/ha, Phosphorus: {main_result['phosphorus']:.2f} kg/ha, and Potassium: {main_result['potassium']:.2f} kg/ha."
        else:
            # Multiple results - summarize
            states = set([d['metadata']['state'] for d in soil_data])
            avg_pH = np.mean([d['metadata']['pH'] for d in soil_data])
            avg_oc = np.mean([d['metadata']['organic_carbon'] for d in soil_data])
            
            return f"Found soil health data across {len(states)} states. Average pH: {avg_pH:.2f}, Average Organic Carbon: {avg_oc:.2f}%."

