'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Phone, PhoneOff, Mic, MicOff, Volume2, VolumeX, Loader2, Heart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

interface TranscriptEntry {
    speaker: 'ai' | 'user';
    text: string;
    timestamp: string;
}

interface VoiceCallModalProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    vehicleId: string;
    alertType?: string;
    alertData?: {
        brake_efficiency?: number;
        component?: string;
        severity?: string;
    };
    ownerName?: string;
    onBookingConfirmed?: (booking: any) => void;
}

export function VoiceCallModal({
    open,
    onOpenChange,
    vehicleId,
    alertType = 'brake_fade',
    alertData = {},
    ownerName = 'Alex',
    onBookingConfirmed
}: VoiceCallModalProps) {
    // Call state
    const [callState, setCallState] = useState<'idle' | 'ringing' | 'connected' | 'ended'>('idle');
    const [callId, setCallId] = useState<string | null>(null);
    const [stage, setStage] = useState<string>('greeting');

    // Audio state
    const [isMuted, setIsMuted] = useState(false);
    const [isSpeakerOn, setIsSpeakerOn] = useState(true);
    const [isListening, setIsListening] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);

    // Transcript
    const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
    const [currentAiMessage, setCurrentAiMessage] = useState<string>('');
    const [booking, setBooking] = useState<any>(null);
    const [textInput, setTextInput] = useState<string>('');  // Fallback text input
    const [detectedEmotion, setDetectedEmotion] = useState<string>('neutral');  // Scene 2: Emotion tracking

    // Web Speech API refs
    const recognitionRef = useRef<any>(null);
    const synthesisRef = useRef<SpeechSynthesisUtterance | null>(null);
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const transcriptEndRef = useRef<HTMLDivElement>(null);

    // Initialize Web Speech API
    useEffect(() => {
        if (typeof window !== 'undefined') {
            // Check for Web Speech API support
            const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
            if (SpeechRecognition) {
                recognitionRef.current = new SpeechRecognition();
                recognitionRef.current.continuous = false;
                recognitionRef.current.interimResults = true;
                recognitionRef.current.lang = 'en-US';

                recognitionRef.current.onresult = (event: any) => {
                    const lastResult = event.results[event.results.length - 1];
                    if (lastResult.isFinal) {
                        const text = lastResult[0].transcript;
                        handleUserInput(text);
                    }
                };

                recognitionRef.current.onend = () => {
                    setIsListening(false);
                };

                recognitionRef.current.onerror = (event: any) => {
                    console.error('Speech recognition error:', event.error);
                    setIsListening(false);
                };
            }
        }

        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.abort();
            }
            if (window.speechSynthesis) {
                window.speechSynthesis.cancel();
            }
        };
    }, []);

    // Auto-scroll transcript
    useEffect(() => {
        transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [transcript]);

    // Initiate call when modal opens
    useEffect(() => {
        if (open && callState === 'idle') {
            initiateCall();
        }
    }, [open]);

    // Play TTS using Web Speech API
    const speakText = useCallback((text: string, audioBase64?: string) => {
        if (!isSpeakerOn) return;

        // Try to use audio base64 first (from server gTTS)
        if (audioBase64 && audioRef.current) {
            const audioBlob = base64ToBlob(audioBase64, 'audio/mp3');
            const audioUrl = URL.createObjectURL(audioBlob);
            audioRef.current.src = audioUrl;
            audioRef.current.play().catch(console.error);
            return;
        }

        // Fall back to Web Speech API
        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;

            // Try to find a good voice
            const voices = window.speechSynthesis.getVoices();
            const preferredVoice = voices.find(v =>
                v.name.includes('Google') || v.name.includes('Microsoft') || v.lang === 'en-US'
            );
            if (preferredVoice) {
                utterance.voice = preferredVoice;
            }

            synthesisRef.current = utterance;
            window.speechSynthesis.speak(utterance);
        }
    }, [isSpeakerOn]);

    // Initiate the call
    const initiateCall = async () => {
        setCallState('ringing');

        try {
            const response = await fetch(`${API_BASE}/voice/${vehicleId}/initiate-call`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    alert_type: alertType,
                    owner_name: ownerName,
                    brake_efficiency: alertData.brake_efficiency || 15
                })
            });

            const data = await response.json();
            if (data.success) {
                setCallId(data.call_id);
                // Simulate ringing for 2 seconds
                setTimeout(() => answerCall(data.call_id), 2000);
            }
        } catch (error) {
            console.error('Failed to initiate call:', error);
            setCallState('idle');
        }
    };

    // Answer the call
    const answerCall = async (id: string) => {
        setCallState('connected');

        try {
            const response = await fetch(`${API_BASE}/voice/${id}/answer`, {
                method: 'POST'
            });

            const data = await response.json();
            if (data.success) {
                setStage(data.stage);
                setCurrentAiMessage(data.message);
                setTranscript(prev => [...prev, {
                    speaker: 'ai',
                    text: data.message,
                    timestamp: new Date().toISOString()
                }]);

                // Speak the greeting
                speakText(data.message, data.audio_base64);
            }
        } catch (error) {
            console.error('Failed to answer call:', error);
        }
    };

    // Handle user voice input
    const handleUserInput = async (text: string) => {
        if (!callId || !text.trim()) return;

        setIsListening(false);
        setIsProcessing(true);

        // Add user message to transcript
        setTranscript(prev => [...prev, {
            speaker: 'user',
            text: text,
            timestamp: new Date().toISOString()
        }]);

        try {
            const response = await fetch(`${API_BASE}/voice/${callId}/input`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_text: text,
                    detected_intent: null
                })
            });

            const data = await response.json();
            if (data.success) {
                setStage(data.stage);
                setCurrentAiMessage(data.message);

                // Add AI response to transcript
                setTranscript(prev => [...prev, {
                    speaker: 'ai',
                    text: data.message,
                    timestamp: new Date().toISOString()
                }]);

                // Handle booking if created
                if (data.booking) {
                    setBooking(data.booking);
                    onBookingConfirmed?.(data.booking);
                }

                // Handle emotion detection (Scene 2 demo)
                if (data.detected_emotion) {
                    setDetectedEmotion(data.detected_emotion);
                } else if (text.toLowerCase().includes('scary') || text.toLowerCase().includes('worried') || text.toLowerCase().includes('afraid')) {
                    setDetectedEmotion('anxious');
                } else if (text.toLowerCase().includes('frustrated') || text.toLowerCase().includes('angry')) {
                    setDetectedEmotion('frustrated');
                } else if (text.toLowerCase().includes('thanks') || text.toLowerCase().includes('great')) {
                    setDetectedEmotion('happy');
                }

                // Speak the response
                speakText(data.message, data.audio_base64);

                // Check if call ended
                if (data.call_ended) {
                    setTimeout(() => {
                        setCallState('ended');
                    }, 3000);
                }
            }
        } catch (error) {
            console.error('Failed to process input:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    // Start listening for voice input
    const startListening = () => {
        if (!recognitionRef.current || isProcessing) return;

        try {
            // Abort any existing session first
            recognitionRef.current.abort();
        } catch {
            // Ignore abort errors
        }

        // Small delay to ensure previous session is fully stopped
        setTimeout(() => {
            if (recognitionRef.current && !isProcessing) {
                try {
                    recognitionRef.current.start();
                    setIsListening(true);
                } catch (error) {
                    console.error('Failed to start listening:', error);
                    setIsListening(false);
                }
            }
        }, 100);
    };

    // Stop listening
    const stopListening = () => {
        if (recognitionRef.current) {
            try {
                recognitionRef.current.stop();
            } catch {
                // Ignore stop errors
            }
            setIsListening(false);
        }
    };

    // End the call
    const endCall = async () => {
        if (callId) {
            try {
                await fetch(`${API_BASE}/voice/${callId}/end`, { method: 'POST' });
            } catch (error) {
                console.error('Failed to end call:', error);
            }
        }

        setCallState('ended');
        setTimeout(() => {
            onOpenChange(false);
            resetState();
        }, 1500);
    };

    // Reset state
    const resetState = () => {
        setCallState('idle');
        setCallId(null);
        setStage('greeting');
        setTranscript([]);
        setCurrentAiMessage('');
        setBooking(null);
        setIsListening(false);
        setIsProcessing(false);
    };

    // Toggle mute
    const toggleMute = () => {
        setIsMuted(!isMuted);
    };

    // Toggle speaker
    const toggleSpeaker = () => {
        setIsSpeakerOn(!isSpeakerOn);
        if (isSpeakerOn) {
            window.speechSynthesis?.cancel();
        }
    };

    // Utility: Convert base64 to blob
    const base64ToBlob = (base64: string, mimeType: string) => {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    };

    // Get stage display name
    const getStageDisplay = (stage: string) => {
        const stages: Record<string, string> = {
            'greeting': 'Connected',
            'alert_explanation': 'Explaining Alert',
            'safety_check': 'Safety Check',
            'scheduling_offer': 'Offering Service',
            'slot_confirmation': 'Confirming Slot',
            'booking_confirmed': 'Booking Confirmed',
            'farewell': 'Goodbye'
        };
        return stages[stage] || stage;
    };

    // Calculate progress
    const getProgress = () => {
        const stages = ['greeting', 'alert_explanation', 'safety_check', 'scheduling_offer', 'slot_confirmation', 'booking_confirmed', 'farewell'];
        const index = stages.indexOf(stage);
        return ((index + 1) / stages.length) * 100;
    }; return (
        <>
            {/* Hidden audio element for server TTS */}
            <audio ref={audioRef} className="hidden" />

            <Dialog open={open} onOpenChange={(isOpen) => {
                if (!isOpen) {
                    endCall();
                }
                onOpenChange(isOpen);
            }}>
                <DialogContent className="sm:max-w-[500px] bg-gradient-to-b from-slate-900 to-slate-950 border-slate-700">
                    <DialogHeader>
                        <DialogTitle className="flex items-center gap-2 text-white">
                            <Phone className="h-5 w-5 text-green-500" />
                            SentinEV Voice Call
                        </DialogTitle>
                        <DialogDescription className="text-slate-400">
                            {callState === 'ringing' && 'Connecting to SentinEV AI...'}
                            {callState === 'connected' && `Active Call - ${getStageDisplay(stage)}`}
                            {callState === 'ended' && 'Call Ended'}
                        </DialogDescription>
                    </DialogHeader>

                    {/* Call Status */}
                    <div className="flex flex-col items-center py-6">
                        {/* Avatar/Status Icon */}
                        <div className={cn(
                            "w-24 h-24 rounded-full flex items-center justify-center mb-4 transition-all",
                            callState === 'ringing' && "bg-yellow-500/20 animate-pulse",
                            callState === 'connected' && "bg-green-500/20",
                            callState === 'ended' && "bg-red-500/20"
                        )}>
                            {callState === 'ringing' && (
                                <Phone className="h-12 w-12 text-yellow-500 animate-bounce" />
                            )}
                            {callState === 'connected' && (
                                <div className="relative">
                                    <Volume2 className="h-12 w-12 text-green-500" />
                                    {isListening && (
                                        <div className="absolute -top-2 -right-2">
                                            <span className="flex h-4 w-4">
                                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                                                <span className="relative inline-flex rounded-full h-4 w-4 bg-red-500"></span>
                                            </span>
                                        </div>
                                    )}
                                </div>
                            )}
                            {callState === 'ended' && (
                                <PhoneOff className="h-12 w-12 text-red-500" />
                            )}
                        </div>

                        {/* Caller Info */}
                        <h3 className="text-xl font-semibold text-white mb-1">SentinEV AI Assistant</h3>
                        <p className="text-slate-400 text-sm">
                            {callState === 'ringing' && 'Incoming call...'}
                            {callState === 'connected' && `Vehicle: ${vehicleId}`}
                            {callState === 'ended' && 'Thank you for using SentinEV'}
                        </p>

                        {/* Progress */}
                        {callState === 'connected' && (
                            <div className="w-full mt-4 px-4">
                                <Progress value={getProgress()} className="h-1" />
                            </div>
                        )}

                        {/* Emotion Badge - Scene 2 Demo */}
                        {callState === 'connected' && detectedEmotion !== 'neutral' && (
                            <div className="mt-3 flex items-center gap-2">
                                <Badge
                                    variant={detectedEmotion === 'anxious' ? 'destructive' :
                                        detectedEmotion === 'frustrated' ? 'destructive' :
                                            'default'}
                                    className="animate-pulse"
                                >
                                    <Heart className="h-3 w-3 mr-1" />
                                    {detectedEmotion === 'anxious' && '😟 Feeling Anxious'}
                                    {detectedEmotion === 'frustrated' && '😤 Frustrated'}
                                    {detectedEmotion === 'happy' && '😊 Happy'}
                                </Badge>
                                <span className="text-xs text-muted-foreground">Emotion adapted</span>
                            </div>
                        )}
                    </div>

                    {/* Transcript */}
                    {callState === 'connected' && (
                        <div className="bg-slate-800/50 rounded-lg p-4 max-h-[200px] overflow-y-auto">
                            <div className="space-y-3">
                                {transcript.map((entry, index) => (
                                    <div
                                        key={index}
                                        className={cn(
                                            "flex",
                                            entry.speaker === 'user' ? 'justify-end' : 'justify-start'
                                        )}
                                    >
                                        <div className={cn(
                                            "max-w-[80%] rounded-lg px-3 py-2",
                                            entry.speaker === 'user'
                                                ? 'bg-blue-600 text-white'
                                                : 'bg-slate-700 text-slate-200'
                                        )}>
                                            <p className="text-sm">{entry.text}</p>
                                        </div>
                                    </div>
                                ))}
                                <div ref={transcriptEndRef} />
                            </div>

                            {isProcessing && (
                                <div className="flex items-center gap-2 mt-2 text-slate-400">
                                    <Loader2 className="h-4 w-4 animate-spin" />
                                    <span className="text-sm">Processing...</span>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Booking Confirmation */}
                    {booking && (
                        <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-3 mt-2">
                            <Badge className="bg-green-500 mb-2">Appointment Booked</Badge>
                            <p className="text-sm text-green-200">
                                {booking.center_name} at {booking.time}
                            </p>
                        </div>
                    )}

                    {/* Control Buttons */}
                    <div className="flex items-center justify-center gap-4 pt-4 border-t border-slate-700">
                        {callState === 'connected' && (
                            <>
                                {/* Mute Button */}
                                <Button
                                    variant="outline"
                                    size="icon"
                                    className={cn(
                                        "rounded-full w-14 h-14",
                                        isMuted && "bg-red-500/20 border-red-500"
                                    )}
                                    onClick={toggleMute}
                                >
                                    {isMuted ? (
                                        <MicOff className="h-6 w-6 text-red-500" />
                                    ) : (
                                        <Mic className="h-6 w-6" />
                                    )}
                                </Button>

                                {/* Push to Talk Button */}
                                <Button
                                    variant={isListening ? "default" : "outline"}
                                    size="icon"
                                    className={cn(
                                        "rounded-full w-20 h-20",
                                        isListening && "bg-red-500 border-red-500 animate-pulse"
                                    )}
                                    onMouseDown={startListening}
                                    onMouseUp={stopListening}
                                    onTouchStart={startListening}
                                    onTouchEnd={stopListening}
                                    disabled={isMuted || isProcessing}
                                >
                                    <Mic className={cn(
                                        "h-8 w-8",
                                        isListening ? "text-white" : "text-slate-400"
                                    )} />
                                </Button>

                                {/* Speaker Button */}
                                <Button
                                    variant="outline"
                                    size="icon"
                                    className={cn(
                                        "rounded-full w-14 h-14",
                                        !isSpeakerOn && "bg-red-500/20 border-red-500"
                                    )}
                                    onClick={toggleSpeaker}
                                >
                                    {isSpeakerOn ? (
                                        <Volume2 className="h-6 w-6" />
                                    ) : (
                                        <VolumeX className="h-6 w-6 text-red-500" />
                                    )}
                                </Button>
                            </>
                        )}

                        {/* End Call Button */}
                        {(callState === 'ringing' || callState === 'connected') && (
                            <Button
                                variant="destructive"
                                size="icon"
                                className="rounded-full w-14 h-14 ml-4"
                                onClick={endCall}
                            >
                                <PhoneOff className="h-6 w-6" />
                            </Button>
                        )}
                    </div>

                    {/* Instructions */}
                    {callState === 'connected' && !isListening && (
                        <p className="text-center text-xs text-slate-500 mt-2">
                            Hold the microphone button and speak, or type below
                        </p>
                    )}
                    {isListening && (
                        <p className="text-center text-xs text-red-400 mt-2 animate-pulse">
                            🎤 Listening... Release to send
                        </p>
                    )}

                    {/* Text Input Fallback */}
                    {callState === 'connected' && (
                        <div className="flex gap-2 mt-3 px-2">
                            <input
                                type="text"
                                value={textInput}
                                onChange={(e) => setTextInput(e.target.value)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter' && textInput.trim() && !isProcessing) {
                                        handleUserInput(textInput.trim());
                                        setTextInput('');
                                    }
                                }}
                                placeholder="Type your response here..."
                                className="flex-1 px-3 py-2 text-sm bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:border-blue-500"
                                disabled={isProcessing}
                            />
                            <Button
                                size="sm"
                                onClick={() => {
                                    if (textInput.trim() && !isProcessing) {
                                        handleUserInput(textInput.trim());
                                        setTextInput('');
                                    }
                                }}
                                disabled={!textInput.trim() || isProcessing}
                            >
                                Send
                            </Button>
                        </div>
                    )}
                </DialogContent>
            </Dialog>
        </>
    );
}
