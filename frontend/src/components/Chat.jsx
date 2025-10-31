import { useState, useEffect, useRef } from 'react'
import { Send, Plus, Menu, Settings, MessageSquare, Loader2 } from 'lucide-react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

const API_URL = 'http://localhost:5000'

const Chat = () => {
  const [messages, setMessages] = useState([
    { 
      role: 'assistant', 
      content: '🌾 Hi, I\'m **SaarthiAI** — your agriculture guide.\n\nI can help with crop/soil data and general farming advice.\n\nTry: "Rice production in Andhra Pradesh?" or "How to improve soil health?"',
      loading: false
    }
  ])
  const [input, setInput] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [conversations] = useState([
    { id: 1, title: 'New Conversation' },
  ])

  // Ref for messages container for auto-scroll
  const messagesEndRef = useRef(null)
  const messagesContainerRef = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  // Auto-scroll to bottom when loading state changes
  useEffect(() => {
    if (isLoading && messagesEndRef.current) {
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }, 100)
    }
  }, [isLoading])

  const handleSend = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    const userMessageObj = { role: 'user', content: userMessage }
    setMessages(prev => [...prev, userMessageObj])
    setInput('')
    setIsLoading(true)

    try {
      const response = await axios.post(`${API_URL}/query`, {
        question: userMessage,
        top_k: 10,  // Get more results for better accuracy
        use_gemini: true  // Enable Gemini enhancement
      })

      const answer = response.data.answer || 'I received an empty response.'
      const assistantMessage = { 
        role: 'assistant', 
        content: answer,
        sources: response.data.sources || [],
        confidence: response.data.confidence || 0,
        ai_enhanced: response.data.ai_enhanced || false
      }
      
      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Error fetching response:', error)
      const errorMessage = error.response?.data?.error || 'Failed to get response from the server. Please make sure the backend is running.'
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `Error: ${errorMessage}`
      }])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex h-screen bg-gradient-to-br from-violet-50 via-purple-50 to-fuchsia-50 text-gray-900">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-white/80 backdrop-blur-xl border-r border-white/20 shadow-xl overflow-hidden`}>
        <div className="p-4 space-y-3 h-full flex flex-col">
          {/* New Chat Button */}
          <button 
            onClick={() => {
              setMessages([{ 
                role: 'assistant', 
                content: '🌾 Hi, I\'m **SaarthiAI** — your agriculture guide.\n\nI can help with crop/soil data and general farming advice.\n\nTry: "Rice production in Andhra Pradesh?" or "How to improve soil health?"',
                loading: false
              }])
            }}
            className="flex items-center gap-3 p-3 rounded-2xl hover:bg-gradient-to-r from-purple-100 to-pink-100 transition-all duration-300 border border-purple-200/50 text-purple-700 w-full font-medium shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98]"
          >
            <Plus size={16} />
            <span className="text-sm font-semibold">New Chat</span>
          </button>

          {/* Conversations List */}
          <div className="flex-1 overflow-y-auto space-y-2">
            {conversations.map(conv => (
              <button
                key={conv.id}
                className="flex items-center gap-3 p-3 rounded-2xl hover:bg-white/70 transition-all duration-300 w-full text-left text-gray-600 hover:text-gray-800 shadow-md hover:shadow-lg"
              >
                <MessageSquare size={16} />
                <span className="text-sm truncate font-medium">{conv.title}</span>
              </button>
            ))}
          </div>

          {/* Bottom Section */}
          <div className="pt-3 border-t border-white/30">
            <button className="flex items-center gap-3 p-3 rounded-2xl hover:bg-white/70 transition-all duration-300 w-full text-gray-600 hover:text-gray-800 shadow-md hover:shadow-lg">
              <Settings size={16} />
              <span className="text-sm font-medium">Settings</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Header */}
        <div className="px-6 py-4 border-b border-white/20 bg-white/60 backdrop-blur-xl shadow-lg flex items-center gap-4">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2.5 hover:bg-white/80 rounded-2xl transition-all duration-300 text-gray-700 hover:shadow-lg active:scale-95"
          >
            <Menu size={20} />
          </button>
          <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 via-pink-500 to-orange-500 bg-clip-text text-transparent">SaarthiAI</h1>
        </div>

        {/* Messages */}
        <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex gap-4 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-sm font-bold text-white shadow-xl">
                    AI
                  </div>
                </div>
              )}
              <div
                className={`max-w-3xl rounded-3xl px-6 py-4 ${
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-blue-500 to-purple-600 text-white shadow-2xl'
                    : 'bg-white/80 backdrop-blur-xl text-gray-800 shadow-2xl border border-white/50'
                }`}
              >
                <div className="text-sm">
                  <ReactMarkdown 
                    remarkPlugins={[remarkGfm]}
                    className="leading-relaxed"
                    components={{
                      // Style code blocks
                      code: ({ node, inline, ...props }) => {
                        if (inline) {
                          return <span className="bg-purple-100 px-2 py-1 rounded-lg text-sm font-mono text-purple-800 border border-purple-200" {...props} />
                        }
                        return <code className="block bg-purple-50 p-3 rounded-2xl my-2 font-mono text-sm overflow-x-auto border border-purple-200 shadow-inner" {...props} />
                      },
                      // Style links
                      a: ({ node, ...props }) => <a className="text-purple-600 hover:text-purple-700 underline font-medium" target="_blank" rel="noopener noreferrer" {...props} />,
                      // Style lists
                      ul: ({ node, ...props }) => <ul className="list-disc space-y-1 my-1" {...props} />,
                      ol: ({ node, ...props }) => <ol className="list-decimal space-y-1 my-1" {...props} />,
                      li: ({ node, ...props }) => <li className="ml-6" {...props} />,
                      // Style headings
                      h1: ({ node, ...props }) => <h1 className="text-2xl font-bold my-2" {...props} />,
                      h2: ({ node, ...props }) => <h2 className="text-xl font-bold my-1.5" {...props} />,
                      h3: ({ node, ...props }) => <h3 className="text-lg font-semibold my-1.5" {...props} />,
                      // Style paragraphs
                      p: ({ node, ...props }) => <p className="mb-1" {...props} />,
                      // Style bold
                      strong: ({ node, ...props }) => <strong className="font-bold" {...props} />,
                      // Style italic
                      em: ({ node, ...props }) => <em className="italic" {...props} />,
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                </div>
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-xs font-bold text-gray-600 mb-3 flex items-center gap-2">
                      <span className="w-2.5 h-2.5 bg-purple-500 rounded-full shadow-sm"></span>
                      Sources & References:
                    </p>
                    <div className="space-y-2">
                      {message.sources.slice(0, 3).map((source, idx) => (
                        <div key={idx} className="text-xs text-gray-600 bg-purple-50 rounded-xl px-3 py-2 inline-block mr-2 border border-purple-100 shadow-sm">
                          {source.dataset} - {source.relevance}
                        </div>
                      ))}
                    </div>
                    {message.confidence > 0 && (
                      <div className="mt-3 text-xs text-gray-500 font-medium">
                        Confidence: {Math.round(message.confidence * 100)}%
                      </div>
                    )}
                  </div>
                )}
              </div>
              {message.role === 'user' && (
                <div className="flex-shrink-0">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-sm font-bold text-white shadow-xl">
                    U
                  </div>
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4 justify-start">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center text-sm font-bold text-white shadow-xl">
                  AI
                </div>
              </div>
              <div className="max-w-3xl rounded-3xl px-5 py-4 bg-white/80 backdrop-blur-xl border border-white/50 shadow-2xl">
                <div className="flex items-center gap-3 text-gray-700">
                  <Loader2 size={18} className="animate-spin text-purple-500" />
                  <span className="font-medium">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          {/* Scroll target for auto-scroll */}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t border-white/20 bg-white/60 backdrop-blur-xl p-6 shadow-2xl">
          <form onSubmit={handleSend} className="max-w-3xl mx-auto">
            <div className="relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSend(e)
                  }
                }}
                placeholder="Ask about crop/soil data or farming advice..."
                className="w-full bg-white/80 border-2 border-white/50 rounded-3xl px-5 py-4 pr-16 resize-none focus:outline-none focus:ring-4 focus:ring-purple-200/50 focus:border-purple-400 text-gray-900 placeholder:text-gray-400 shadow-lg backdrop-blur-xl transition-all duration-300"
                rows="1"
                disabled={isLoading}
              />
              <button
                type="submit"
                disabled={isLoading}
                className="absolute right-3 bottom-3 p-2.5 bg-gradient-to-br from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-2xl transition-all duration-300 shadow-xl hover:shadow-2xl hover:scale-105 active:scale-95"
              >
                <Send size={18} className="text-white" />
              </button>
            </div>
            <p className="text-xs text-gray-500 mt-3 text-center font-medium">
              ✨ SaarthiAI — Ask data or farming questions
            </p>
          </form>
        </div>
      </div>
    </div>
  )
}

export default Chat

