"""
Flask API Server for Intelligent Q&A System
Provides REST API endpoint for answering questions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
import pickle

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
    """Home endpoint"""
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
    """Health check endpoint"""
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
    """Handle Q&A queries"""
    try:
        qa = init_qa_system()
        
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'error': 'Missing question parameter'
            }), 400
        
        question = data['question']
        top_k = data.get('top_k', 10)  # Increased default to get more results
        use_gemini = data.get('use_gemini', True)  # Gemini enabled by default
        
        print(f"\n🔍 Received question: {question}")
        
        # Get answer from Q&A system
        result = qa.answer_question(question, top_k=top_k)
        print(f"✅ Q&A system returned answer with {result.get('search_results_count', 0)} results")
        
        # Enhance with Gemini if available and requested (for both dataset questions AND general questions)
        if GEMINI_AVAILABLE and use_gemini:
            try:
                print("🤖 Attempting to enhance with Gemini...")
                # Always try Gemini enhancement - it can handle both data-specific and general questions
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
                if enhanced_result.get('general_guidance', False):
                    print("✅ Response provided as general guidance (no dataset match)")
                else:
                    print("✅ Response enhanced with Gemini using dataset")
            except Exception as e:
                print(f"⚠️ Gemini enhancement failed: {e}")
                # Use basic response
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
            # Use basic response without Gemini (Gemini disabled)
            response = {
                'question': question,
                'answer': result['answer'],
                'confidence': result['confidence'],
                'sources': result.get('sources', []),
                'num_results': result.get('search_results_count', 0),
                'ai_enhanced': False
            }
            print("✅ Using basic Q&A response (Gemini disabled)")
        
        return jsonify(response), 200
        
    except Exception as e:
        import traceback
        print(f"❌ Error in query endpoint: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'details': traceback.format_exc()
        }), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get system statistics"""
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
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("Starting Q&A API Server...")
    print("=" * 80)
    print("\n📡 API Endpoints:")
    print("  - GET  /          : API information")
    print("  - GET  /health     : Health check")
    print("  - GET  /stats      : System statistics")
    print("  - POST /query      : Query the Q&A system")
    print(f"\n🤖 Gemini AI: {'Available' if GEMINI_AVAILABLE else 'Not available'}")
    print("\n🚀 Server will start on: http://localhost:5000")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
