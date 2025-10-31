"""
Gemini AI Service for Enhanced Answer Generation
Uses Google's Gemini API to generate natural, conversational responses
"""

try:
    import google.generativeai as genai
    import config
    
    # Configure Gemini
    genai.configure(api_key=config.GEMINI_API_KEY)
    
    # Initialize the model (using stable 2.5-flash model)
    model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_READY = True
    print("Gemini model 'gemini-2.5-flash' initialized successfully")
except Exception as e:
    print(f"Warning: Gemini not available: {e}")
    GEMINI_READY = False
    model = None

def generate_smart_response(user_question, retrieved_data):
    """
    Generate a natural, conversational response using Gemini AI
    based on the retrieved data from the Q&A system
    
    Args:
        user_question: The user's original question
        retrieved_data: Dictionary containing retrieved chunks and sources
    
    Returns:
        Enhanced natural language response
    """
    
    # Check if Gemini is available
    if not GEMINI_READY or model is None:
        raise Exception("Gemini not available")
    
    # Even if no specific data found, we can still answer general questions using Gemini's knowledge
    # So we'll proceed with the generation regardless of data availability
    has_data = retrieved_data and retrieved_data.get('sources') and len(retrieved_data['sources']) > 0
    
    # Prepare context for Gemini
    context = f"""You are **SaarthiAI**, an expert agriculture assistant and guide helping users with questions about Indian agriculture, crop production, soil health, and general agricultural practices.

Your role is to:
- Answer questions about Indian agriculture using available data
- Guide users with general agricultural knowledge and best practices
- Provide helpful advice on farming, crops, soil management, and agricultural techniques
- Act as a knowledgeable mentor and guide for farmers and agriculture enthusiasts

User Question: {user_question}

Retrieved Data from Knowledge Base:
"""
    
    # Add retrieved chunks if available
    if retrieved_data.get('sources') and len(retrieved_data['sources']) > 0:
        for i, source in enumerate(retrieved_data['sources'][:3], 1):
            context += f"\nSource {i}:\n"
            context += f"Dataset: {source['dataset']}\n"
            context += f"Information: {source['chunk']}\n"
            if 'details' in source:
                context += f"Details: {source['details']}\n"
            context += f"Relevance: {source['relevance']}\n"
    else:
        context += "\nNote: No specific dataset matches found for this question. Please provide general guidance based on your agricultural expertise.\n"
    
    # Create the prompt
    prompt = f"""{context}

Based on the information above (if available), provide a helpful, clear, and conversational answer to the user's question. 

Guidelines:
1. **Always introduce yourself as SaarthiAI** at the beginning if this is a general question
2. Answer naturally and conversationally, acting as a helpful guide
3. If specific data is provided, use it and cite it appropriately
4. If no specific data is available, use your agricultural expertise to provide general guidance
5. For general agriculture questions, provide comprehensive guidance including:
   - Best practices and recommendations
   - Agricultural techniques and methods
   - Crop management advice
   - Soil health improvement tips
   - Common agricultural knowledge
6. Be encouraging and supportive, like a mentor
7. If applicable, mention that you can also provide specific data queries if they have questions about particular regions or crops
8. Include relevant numbers and statistics when data is available
9. If multiple sources provide data, synthesize them intelligently

Your Response:"""

    try:
        # Generate response using Gemini
        response = model.generate_content(prompt)
        
        enhanced_answer = response.text.strip()
        
        return {
            'answer': enhanced_answer,
            'confidence': retrieved_data.get('confidence', 0) if has_data else 0.5,  # Default confidence for general answers
            'sources': retrieved_data.get('sources', []),
            'ai_enhanced': True,
            'general_guidance': not has_data  # Flag to indicate this is general guidance
        }
        
    except Exception as e:
        print(f"Gemini API Error: {str(e)}")
        # Fallback to original response or provide a basic message
        fallback_answer = retrieved_data.get('answer', "")
        if not fallback_answer or fallback_answer.strip() == "":
            fallback_answer = "I'm **SaarthiAI**, your agriculture assistant. I apologize, but I'm having trouble processing your question right now. Please try asking about:\n\n- Crop production data\n- Soil health information\n- General agriculture guidance\n- Farming best practices"
        return {
            'answer': fallback_answer,
            'confidence': retrieved_data.get('confidence', 0),
            'sources': retrieved_data.get('sources', []),
            'ai_enhanced': False,
            'fallback': True
        }

def check_gemini_connection():
    """Check if Gemini API is working"""
    try:
        if not GEMINI_READY or model is None:
            return False, "Gemini not initialized"
        test_response = model.generate_content("Say 'Hello' if you're working.")
        return True, test_response.text
    except Exception as e:
        return False, str(e)

