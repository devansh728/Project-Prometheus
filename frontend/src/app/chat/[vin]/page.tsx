// AI Chatbot Page

'use client';

import { useState, useRef, useEffect } from 'react';
import { useParams, useSearchParams } from 'next/navigation';
import { Send, Bot, User, Loader2, ArrowLeft, Sparkles, Calendar } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Navbar } from '@/components/layout/Navbar';
import { Sidebar } from '@/components/layout/Sidebar';
import { useChatStore } from '@/stores/chatStore';
import { sendChatMessage, startSchedulingConversation } from '@/lib/api';
import { toast } from 'sonner';
import { cn } from '@/lib/utils';
import Link from 'next/link';
import type { ChatMessage } from '@/types';

const QUICK_ACTIONS = [
    { label: 'Vehicle Status', message: 'What is the current status of my vehicle?' },
    { label: 'Battery Health', message: 'How is my battery health?' },
    { label: 'Driving Tips', message: 'Give me some driving tips to improve efficiency' },
    { label: 'Recent Alerts', message: 'What are the recent alerts for my vehicle?' },
    { label: 'Maintenance', message: 'When should I schedule my next maintenance?' },
    { label: '📅 Schedule Service', message: 'I want to schedule a service appointment' },
];

export default function ChatPage() {
    const params = useParams();
    const searchParams = useSearchParams();
    const vin = params.vin as string;

    const [sidebarOpen, setSidebarOpen] = useState(false);
    const [input, setInput] = useState('');
    const [schedulingSlots, setSchedulingSlots] = useState<any[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    const { messagesByVehicle, addMessage, isTyping, setTyping, sessionId, setSessionId } = useChatStore();
    const messages = messagesByVehicle[vin] || [];

    // Auto-start scheduling conversation if redirected from diagnosis
    useEffect(() => {
        if (searchParams.get('auto') === 'true' && messages.length === 0) {
            async function autoStartScheduling() {
                setTyping(true);
                try {
                    const response = await startSchedulingConversation(vin) as {
                        has_context?: boolean;
                        intro_message?: string;
                        scheduling_message?: string;
                        slots?: any[];
                    };

                    if (response.has_context && response.intro_message) {
                        // Add intro message
                        const introMsg: ChatMessage = {
                            id: `ai-intro-${Date.now()}`,
                            role: 'assistant',
                            content: response.intro_message,
                            timestamp: new Date().toISOString(),
                        };
                        addMessage(vin, introMsg);

                        // If there are slots, add scheduling message
                        if (response.scheduling_message) {
                            setTimeout(() => {
                                const schedMsg: ChatMessage = {
                                    id: `ai-sched-${Date.now()}`,
                                    role: 'assistant',
                                    content: response.scheduling_message!,
                                    timestamp: new Date().toISOString(),
                                };
                                addMessage(vin, schedMsg);
                            }, 1000);
                        }

                        // Store slots for quick action buttons
                        if (response.slots) {
                            setSchedulingSlots(response.slots);
                        }

                        toast.success('📅 Scheduling Assistant Ready!');
                    }
                } catch (error) {
                    console.error('Failed to start scheduling:', error);
                } finally {
                    setTyping(false);
                }
            }
            autoStartScheduling();
        }
    }, [searchParams, vin, addMessage, setTyping, messages.length]);

    // Scroll to bottom on new messages
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, isTyping]);

    // Send message
    async function handleSend() {
        const text = input.trim();
        if (!text || isTyping) return;

        // Add user message
        const userMessage: ChatMessage = {
            id: `user-${Date.now()}`,
            role: 'user',
            content: text,
            timestamp: new Date().toISOString(),
        };
        addMessage(vin, userMessage);
        setInput('');
        setTyping(true);

        try {
            const response = await sendChatMessage(vin, text, sessionId || undefined);

            // Save session ID if new
            if (response.session_id && response.session_id !== sessionId) {
                setSessionId(response.session_id);
            }

            // Add AI response
            const aiMessage: ChatMessage = {
                id: `ai-${Date.now()}`,
                role: 'assistant',
                content: response.response,
                timestamp: response.timestamp,
            };
            addMessage(vin, aiMessage);
        } catch (error) {
            toast.error('Failed to send message');
            console.error(error);
        } finally {
            setTyping(false);
        }
    }

    // Handle quick action
    function handleQuickAction(message: string) {
        setInput(message);
    }

    // Handle enter key
    function handleKeyDown(e: React.KeyboardEvent) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    }

    return (
        <div className="min-h-screen flex flex-col">
            <Navbar onMenuClick={() => setSidebarOpen(true)} />

            <div className="flex flex-1">
                <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />

                <main className="flex-1 flex flex-col h-[calc(100vh-4rem)]">
                    {/* Header */}
                    <div className="p-4 border-b flex items-center gap-4">
                        <Link href={`/vehicles/${vin}`}>
                            <Button variant="ghost" size="icon">
                                <ArrowLeft className="h-4 w-4" />
                            </Button>
                        </Link>
                        <div>
                            <h1 className="text-lg font-semibold flex items-center gap-2">
                                <Bot className="h-5 w-5" />
                                AI Assistant
                            </h1>
                            <p className="text-sm text-muted-foreground">Vehicle: {vin}</p>
                        </div>
                    </div>

                    {/* Chat Area */}
                    <div className="flex-1 flex">
                        {/* Messages */}
                        <div className="flex-1 flex flex-col">
                            <ScrollArea className="flex-1 p-4" ref={scrollRef}>
                                {messages.length === 0 ? (
                                    <div className="h-full flex flex-col items-center justify-center text-center p-4">
                                        <Sparkles className="h-12 w-12 text-muted-foreground mb-4" />
                                        <h2 className="text-xl font-semibold mb-2">Welcome to SentinEV AI</h2>
                                        <p className="text-muted-foreground max-w-md">
                                            I can help you understand your vehicle's health, explain anomalies,
                                            and provide personalized driving recommendations.
                                        </p>
                                    </div>
                                ) : (
                                    <div className="space-y-4 max-w-3xl mx-auto">
                                        {messages.map((msg) => (
                                            <div
                                                key={msg.id}
                                                className={cn(
                                                    'flex gap-3',
                                                    msg.role === 'user' ? 'justify-end' : 'justify-start'
                                                )}
                                            >
                                                {msg.role === 'assistant' && (
                                                    <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center flex-shrink-0">
                                                        <Bot className="h-4 w-4 text-primary-foreground" />
                                                    </div>
                                                )}

                                                <div
                                                    className={cn(
                                                        'max-w-[80%] rounded-lg px-4 py-2',
                                                        msg.role === 'user'
                                                            ? 'bg-primary text-primary-foreground'
                                                            : 'bg-muted'
                                                    )}
                                                >
                                                    <p className="whitespace-pre-wrap">{msg.content}</p>
                                                    <p className="text-xs opacity-60 mt-1">
                                                        {new Date(msg.timestamp).toLocaleTimeString()}
                                                    </p>
                                                </div>

                                                {msg.role === 'user' && (
                                                    <div className="h-8 w-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
                                                        <User className="h-4 w-4" />
                                                    </div>
                                                )}
                                            </div>
                                        ))}

                                        {isTyping && (
                                            <div className="flex gap-3 justify-start">
                                                <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center">
                                                    <Bot className="h-4 w-4 text-primary-foreground" />
                                                </div>
                                                <div className="bg-muted rounded-lg px-4 py-2">
                                                    <Loader2 className="h-4 w-4 animate-spin" />
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </ScrollArea>

                            {/* Input Area */}
                            <div className="border-t p-4">
                                <div className="max-w-3xl mx-auto">
                                    <div className="flex gap-2">
                                        <Input
                                            placeholder="Ask about your vehicle..."
                                            value={input}
                                            onChange={(e) => setInput(e.target.value)}
                                            onKeyDown={handleKeyDown}
                                            disabled={isTyping}
                                            className="flex-1"
                                        />
                                        <Button onClick={handleSend} disabled={!input.trim() || isTyping}>
                                            <Send className="h-4 w-4" />
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Quick Actions Sidebar */}
                        <div className="hidden lg:block w-64 border-l p-4">
                            <h3 className="font-medium mb-4">Quick Actions</h3>
                            <div className="space-y-2">
                                {QUICK_ACTIONS.map((action) => (
                                    <Button
                                        key={action.label}
                                        variant="outline"
                                        size="sm"
                                        className="w-full justify-start text-left"
                                        onClick={() => handleQuickAction(action.message)}
                                    >
                                        {action.label}
                                    </Button>
                                ))}
                            </div>
                        </div>
                    </div>
                </main>
            </div>
        </div>
    );
}
