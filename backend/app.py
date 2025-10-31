import os
import requests
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

# Add parent directory to path to import qa_system
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import qa_system
    from qa_system import IntelligentQASystem
except ImportError:
    print("Warning: Could not import qa_system. Make sure qa_system.py exists.")

# Import Gemini service
try:
    import gemini_service
    from gemini_service import generate_smart_response, check_gemini_connection
    GEMINI_AVAILABLE = True
    print("✅ Gemini AI integration available")
except ImportError as e:
    print(f"⚠️ Warning: Gemini service not available: {e}")
    GEMINI_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Gemini: {e}")
    GEMINI_AVAILABLE = False

# Ensure vector_database.pkl exists by downloading from Google Drive
VECTOR_DB_FILE = "backend/vector_database.pkl"
GOOGLE_DRIVE_FILE_ID = "1t5ZyZ9MFjSVOLZpZR-ZqvPzBBFyt6617"

if not os.path.exists(VECTOR_DB_FILE):
    print("Downloading vector_database.pkl from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    os.makedirs(os.path.dirname(VECTOR_DB_FILE), exist_ok=True)
    with open(VECTOR_DB_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variable to hold Q&A system
qa_system = None

def init_qa_system():
    """Initialize Q&A system lazily"""
    global qa_system
    if qa_system is None:
        try:
            print("Initializing Q&A System...")
            qa_system = IntelligentQASystem()
            print("✅ Q&A System ready!")
        except Exception as e:
            print(f"❌ Error initializing Q&A system: {e}")
            raise e
    return qa_system

@app.route('/')
def home():
    return jsonify({
        'message': 'Intelligent Q&A API Server',
        'status': 'running',
        'gemini_available': GEMINI_AVAILABLE,
        'endpoints': {
            '/query': 'POST - Query the Q&A system',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    try:
        qa = init_qa_system()
        return jsonify({
            'status': 'healthy',
            'message': 'Q&A system is running',
            'gemini_available': GEMINI_AVAILABLE
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        qa = init_qa_system()
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'error': 'Missing question parameter'}), 400
        
        question = data['question']
        top_k = data.get('top_k', 10)
        use_gemini = data.get('use_gemini', True)

        result = qa.answer_question(question, top_k=top_k)
        if GEMINI_AVAILABLE and use_gemini:
            try:
                enhanced_result = generate_smart_response(question, result)
                response = {
                    'question': question,
                    'answer': enhanced_result['answer'],
                    'confidence': enhanced_result['confidence'],
                    'sources': enhanced_result.get('sources', result.get('sources', [])),
                    'num_results': result.get('search_results_count', 0),
                    'ai_enhanced': enhanced_result.get('ai_enhanced', False),
                    'general_guidance': enhanced_result.get('general_guidance', False)
                }
            except Exception:
                response = {
                    'question': question,
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'sources': result.get('sources', []),
                    'num_results': result.get('search_results_count', 0),
                    'ai_enhanced': False,
                    'fallback': True
                }
        else:
            response = {
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources': result.get('sources', []),
                'num_results': result.get('search_results_count', 0),
                'ai_enhanced': False
            }
        return jsonify(response), 200
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'details': traceback.format_exc()}), 500

@app.route('/stats', methods=['GET'])
def stats():
    try:
        qa = init_qa_system()
        return jsonify({
            'total_chunks': len(qa.chunks),
            'method': qa.vector_db['method'],
            'gemini_available': GEMINI_AVAILABLE,
            'datasets': {
                'crop_production': 'soil_health_complete_dataset.csv',
                'soil_health': 'soil_health_complete_dataset.csv'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
