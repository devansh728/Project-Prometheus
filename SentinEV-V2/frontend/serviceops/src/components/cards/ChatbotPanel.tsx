/**
 * Chatbot Panel - What-if query interface for service center heads
 */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { MessageSquare, Send, Loader2, HelpCircle, TrendingUp, User } from 'lucide-react';
import styles from './ChatbotPanel.module.css';

interface ChatMessage {
  type: 'user' | 'bot';
  content: string;
  data?: any;
  suggestions?: string[];
}

interface ChatbotPanelProps {
  centerId?: string;
  onQuery: (query: string, centerId: string) => Promise<any>;
}

export const ChatbotPanel: React.FC<ChatbotPanelProps> = ({
  centerId = 'SC001',
  onQuery,
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      type: 'bot',
      content: "Hi! I'm your ServiceOps AI assistant. Ask me about workload, technician availability, or capacity planning.",
      suggestions: [
        "Which day is least busy next week?",
        "If a technician is on leave, what happens?",
        "Can we accept more brake jobs tomorrow?",
      ],
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSend = async (text?: string) => {
    const query = text || input;
    if (!query.trim()) return;

    // Add user message
    setMessages(prev => [...prev, { type: 'user', content: query }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await onQuery(query, centerId);
      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          content: response.response || "I couldn't process that query.",
          data: response.data || response.impact || response.recommendation,
          suggestions: response.suggestions,
        },
      ]);
    } catch (error) {
      setMessages(prev => [
        ...prev,
        {
          type: 'bot',
          content: "Sorry, there was an error processing your query. Please try again.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSend(suggestion);
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <MessageSquare className={styles.icon} size={20} />
        <div className={styles.titleArea}>
          <h3 className={styles.title}>Operations Chatbot</h3>
          <span className={styles.subtitle}>Ask what-if questions</span>
        </div>
      </div>

      <div className={styles.messages}>
        <AnimatePresence>
          {messages.map((msg, index) => (
            <motion.div
              key={index}
              className={`${styles.message} ${styles[msg.type]}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ delay: index * 0.05 }}
            >
              <div className={styles.avatar}>
                {msg.type === 'bot' ? <HelpCircle size={16} /> : <User size={16} />}
              </div>
              <div className={styles.content}>
                <p>{msg.content}</p>
                
                {msg.data && (
                  <div className={styles.dataBox}>
                    <TrendingUp size={12} />
                    <pre>{JSON.stringify(msg.data, null, 2)}</pre>
                  </div>
                )}

                {msg.suggestions && msg.suggestions.length > 0 && (
                  <div className={styles.suggestions}>
                    {msg.suggestions.map((sug, i) => (
                      <button
                        key={i}
                        className={styles.suggestionChip}
                        onClick={() => handleSuggestionClick(sug)}
                      >
                        {sug}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {isLoading && (
          <motion.div
            className={`${styles.message} ${styles.bot}`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className={styles.avatar}>
              <Loader2 className={styles.spinner} size={16} />
            </div>
            <div className={styles.content}>
              <p>Analyzing...</p>
            </div>
          </motion.div>
        )}
      </div>

      <div className={styles.inputArea}>
        <input
          type="text"
          className={styles.input}
          placeholder="Ask about workload, capacity, or scheduling..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          disabled={isLoading}
        />
        <button
          className={styles.sendButton}
          onClick={() => handleSend()}
          disabled={!input.trim() || isLoading}
        >
          <Send size={18} />
        </button>
      </div>
    </div>
  );
};

export default ChatbotPanel;
