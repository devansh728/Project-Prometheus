/**
 * Brake Fade Voice Call Screen - WebView Speech Recognition Version
 * Works in Expo Go - No native modules required
 */
import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Pressable,
  Animated,
  Dimensions,
  ScrollView,
  Platform,
  Alert,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import * as Speech from 'expo-speech'; // TTS only — for AI speaking
import { SpeechWebView, SpeechWebViewRef } from '../../components/voice/SpeechWebView';
import { fuzzyMatch } from '../../hooks/useWebViewSpeechRecognition';
import { theme } from '../../theme';

const { width, height } = Dimensions.get('window');

const BRAKE_FADE_SCRIPTS = {
  greeting: {
    ai: "Hello! This is your SentinEV assistant calling about your Kia EV6. I've noticed some concerning brake patterns that I'd like to discuss with you.",
    userOptions: ["What's wrong?", 'Is it safe to drive?', "I'm busy right now"],
  },
  explanation: {
    ai: "I've detected increased brake temperature and reduced efficiency in your braking system. Based on our analysis, there's approximately 65% probability of brake fade within the next 6-7 days if not addressed.",
    userOptions: ['How serious is this?', 'What do you recommend?', 'How much will it cost?'],
  },
  seriousness: {
    ai: "It's a preventive concern right now, not an emergency. Your brakes are still functional, but the wear pattern suggests they'll need attention soon. Catching it early means a simple brake pad replacement rather than more extensive repairs.",
    userOptions: ['Okay, what should I do?', 'Can I wait until next week?'],
  },
  recommendation: {
    ai: "I recommend scheduling a brake service within the next 3-5 days. I've already found an optimal slot for you at EV Care Mumbai Central - they have brake specialists available and all necessary parts in stock.",
    userOptions: ['Book it for me', 'What are the available times?', 'Send me details instead'],
  },
  cost: {
    ai: "Based on the current brake pad wear level, the estimated service cost would be around 4,500 to 6,000 rupees for premium brake pad replacement, including labor and inspection. This is covered under your extended warranty agreement.",
    userOptions: ['That sounds reasonable', 'Book the appointment'],
  },
  availableTimes: {
    ai: "I have the following slots available: Tomorrow at 10 AM, Thursday at 2 PM, or Friday at 11 AM. The service will take approximately 2 hours. Which works best for you?",
    userOptions: ['Tomorrow 10 AM works', 'Thursday 2 PM', 'Friday 11 AM'],
  },
  waitQuestion: {
    ai: "While you can technically wait, I wouldn't recommend it beyond 5-6 days. The degradation pattern I'm seeing suggests the wear is accelerating. Addressing it now prevents potential safety issues and costlier repairs.",
    userOptions: ["Okay, let's schedule it", 'Book the earliest slot'],
  },
  safetyResponse: {
    ai: "Yes, your vehicle is safe to drive for now. However, I recommend avoiding aggressive braking and long downhill drives until the service is completed. The warning indicators show you have about 6-7 days before it becomes a concern.",
    userOptions: ['Good to know', 'What should I do?', 'Schedule maintenance'],
  },
  busyResponse: {
    ai: "I understand you're busy. Would you prefer I send you the details via message? You can review them at your convenience and book directly through the app. I just want to make sure you're aware of the brake condition.",
    userOptions: ['Yes, send details', "I'll call back later", 'Quick summary please'],
  },
  booking: {
    ai: "Perfect! I've scheduled your brake service for {time} at EV Care Mumbai Central. You'll receive a confirmation with directions shortly. Is there anything else I can help you with?",
    userOptions: ["No, that's all", 'What should I expect?'],
  },
  serviceExpectation: {
    ai: "When you arrive, just check in at the service desk. The technician will perform a complete brake inspection, replace the worn pads, check brake fluid levels, and calibrate the system. You can wait in their comfortable lounge or I can arrange a pickup and drop service.",
    userOptions: ['Sounds good, thanks!', 'Arrange pickup please'],
  },
  sendDetails: {
    ai: "Done! I've sent you a detailed summary including the diagnosis, recommended service, cost estimate, and available appointment slots. You can book directly from the app whenever you're ready. Take care and drive safely!",
    userOptions: ['Thanks!', 'Got it'],
  },
  farewell: {
    ai: "Great! Your appointment is confirmed. Drive safely, and remember, I'm always here monitoring your vehicle. Have a wonderful day!",
    userOptions: [],
  },
};

interface BrakeFadeVoiceCallProps {
  visible: boolean;
  onClose: () => void;
  onBookingComplete?: (bookingDetails: { time: string; center: string }) => void;
}

type ConversationStep = keyof typeof BRAKE_FADE_SCRIPTS;

export const BrakeFadeVoiceCall: React.FC<BrakeFadeVoiceCallProps> = ({
  visible,
  onClose,
  onBookingComplete,
}) => {
  const [currentStep, setCurrentStep] = useState<ConversationStep>('greeting');
  const [conversationHistory, setConversationHistory] = useState<
    Array<{ role: 'ai' | 'user'; text: string }>
  >([]);
  const [isAISpeaking, setIsAISpeaking] = useState(false);
  const [selectedTime, setSelectedTime] = useState<string>('');
  const [availableSlots, setAvailableSlots] = useState<any[]>([]);
  const [selectedSlot, setSelectedSlot] = useState<any>(null);

  // Voice recognition state
  const [isListening, setIsListening] = useState(false);
  const [interimText, setInterimText] = useState('');
  const [finalText, setFinalText] = useState('');
  const [speechReady, setSpeechReady] = useState(false);
  const [matchFeedback, setMatchFeedback] = useState('');

  const speechWebViewRef = useRef<SpeechWebViewRef>(null);
  const scrollViewRef = useRef<ScrollView>(null);
  const currentStepRef = useRef<ConversationStep>('greeting');

  // Keep ref in sync
  useEffect(() => {
    currentStepRef.current = currentStep;
  }, [currentStep]);

  // Animations
  const [pulseAnim] = useState(new Animated.Value(1));
  const [waveAnim] = useState(new Animated.Value(0));

  useEffect(() => {
    if (visible) {
      setCurrentStep('greeting');
      setConversationHistory([]);
      setInterimText('');
      setFinalText('');

      setTimeout(() => {
        addAIMessage(BRAKE_FADE_SCRIPTS.greeting.ai);
      }, 1500);

      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.15,
            duration: 800,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 800,
            useNativeDriver: true,
          }),
        ])
      ).start();

      Animated.loop(
        Animated.timing(waveAnim, {
          toValue: 1,
          duration: 1500,
          useNativeDriver: true,
        })
      ).start();
    }
  }, [visible]);

  // Auto-scroll to bottom
  useEffect(() => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 200);
  }, [conversationHistory, interimText]);

  // TTS for AI messages
  const addAIMessage = useCallback((text: string) => {
    setIsAISpeaking(true);
    setConversationHistory((prev) => [...prev, { role: 'ai', text }]);

    Speech.speak(text, {
      language: 'en-US',
      pitch: 1.0,
      rate: Platform.OS === 'ios' ? 0.92 : 0.88,
      onDone: () => setIsAISpeaking(false),
      onStopped: () => setIsAISpeaking(false),
      onError: () => setIsAISpeaking(false),
    });
  }, []);

  // Handle speech recognition events
  const handleInterimResult = useCallback((text: string) => {
    setInterimText(text);
  }, []);

  const handleFinalResult = useCallback(
    (text: string, confidence: number) => {
      setFinalText(text);
      setInterimText('');

      console.log(
        `[SPEECH] Final: "${text}" (confidence: ${(confidence * 100).toFixed(1)}%)`
      );

      // Get current options
      const step = currentStepRef.current;
      const currentOptions = BRAKE_FADE_SCRIPTS[step].userOptions;

      if (currentOptions.length === 0) return;

      // Fuzzy match
      const result = fuzzyMatch(text, currentOptions, 0.3);

      if (result.match) {
        console.log(
          `[MATCH] "${text}" → "${result.match}" (score: ${result.score.toFixed(2)})`
        );
        setMatchFeedback(`Matched: "${result.match}"`);

        // Stop listening before responding
        speechWebViewRef.current?.stopListening();
        setIsListening(false);

        // Small delay so user sees the match
        setTimeout(() => {
          setMatchFeedback('');
          setFinalText('');
          handleUserResponse(result.match!);
        }, 1000);
      } else {
        console.log(`[NO MATCH] "${text}" — no option matched`);
        setMatchFeedback('Didn\'t catch that — try again or tap an option');
        setTimeout(() => setMatchFeedback(''), 2500);
      }
    },
    []
  );

  const handleSpeechStatusChange = useCallback((status: string) => {
    console.log('[SPEECH STATUS]', status);
    if (status === 'ready') {
      setSpeechReady(true);
    } else if (status === 'listening') {
      setIsListening(true);
    } else if (status === 'stopped') {
      setIsListening(false);
    }
  }, []);

  const handleSpeechError = useCallback((error: string) => {
    console.log('[SPEECH ERROR]', error);
    if (error === 'not-allowed') {
      Alert.alert(
        'Microphone Permission',
        'Please allow microphone access in your browser/WebView settings to use voice input.',
        [{ text: 'OK' }]
      );
    }
  }, []);

  // Toggle listening
  const toggleListening = useCallback(() => {
    if (isAISpeaking) {
      Speech.stop();
      setIsAISpeaking(false);
    }

    if (isListening) {
      speechWebViewRef.current?.stopListening();
      setIsListening(false);
    } else {
      setInterimText('');
      setFinalText('');
      setMatchFeedback('');
      speechWebViewRef.current?.startListening();
    }

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  }, [isListening, isAISpeaking]);

  // Handle user response (same navigation logic as before)
  const handleUserResponse = useCallback(
    async (response: string) => {
      Speech.stop();
      speechWebViewRef.current?.stopListening();
      setIsListening(false);

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      setConversationHistory((prev) => [...prev, { role: 'user', text: response }]);

      setTimeout(async () => {
        const lower = response.toLowerCase();

        if (lower.includes('wrong') || lower.includes("what's")) {
          setCurrentStep('explanation');
          addAIMessage(BRAKE_FADE_SCRIPTS.explanation.ai);
        } else if (lower.includes('safe') || lower.includes('drive')) {
          setCurrentStep('safetyResponse');
          addAIMessage(BRAKE_FADE_SCRIPTS.safetyResponse.ai);
        } else if (lower.includes('busy')) {
          setCurrentStep('busyResponse');
          addAIMessage(BRAKE_FADE_SCRIPTS.busyResponse.ai);
        } else if (lower.includes('serious')) {
          setCurrentStep('seriousness');
          addAIMessage(BRAKE_FADE_SCRIPTS.seriousness.ai);
        } else if (lower.includes('recommend') || lower.includes('should')) {
          setCurrentStep('recommendation');
          addAIMessage(BRAKE_FADE_SCRIPTS.recommendation.ai);
        } else if (lower.includes('cost') || lower.includes('much')) {
          setCurrentStep('cost');
          addAIMessage(BRAKE_FADE_SCRIPTS.cost.ai);
        } else if (lower.includes('time') || lower.includes('available')) {
          setCurrentStep('availableTimes');

          try {
            const apiUrl =
              process.env.EXPO_PUBLIC_API_URL || 'http://10.79.239.149:8000';
            const resp = await fetch(`${apiUrl}/api/v1/serviceops/find-slots`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                customer_lat: 19.076,
                customer_lon: 72.8777,
                failure_type: 'brake_system',
                severity: 'MEDIUM',
                max_distance_km: 15,
              }),
            });
            const data = await resp.json();
            const slots = data.slots?.slice(0, 3) || [];
            setAvailableSlots(slots);

            if (slots.length > 0) {
              const centerName =
                typeof slots[0].service_center === 'object'
                  ? slots[0].service_center.name
                  : slots[0].service_center;
              const timeStr = new Date(slots[0].start_time).toLocaleTimeString(
                [],
                { hour: '2-digit', minute: '2-digit' }
              );
              const aiText = `I found ${slots.length} excellent options nearby. The best rated is at ${centerName} for tomorrow at ${timeStr}. Would you like to book that?`;
              BRAKE_FADE_SCRIPTS.availableTimes.ai = aiText;
              BRAKE_FADE_SCRIPTS.availableTimes.userOptions = [
                'Yes, book it',
                'Show other times',
                'No thanks',
              ];
              addAIMessage(aiText);
            } else {
              addAIMessage(BRAKE_FADE_SCRIPTS.availableTimes.ai);
            }
          } catch (e) {
            console.error('Failed to fetch slots', e);
            addAIMessage(BRAKE_FADE_SCRIPTS.availableTimes.ai);
          }
        } else if (lower.includes('wait')) {
          setCurrentStep('waitQuestion');
          addAIMessage(BRAKE_FADE_SCRIPTS.waitQuestion.ai);
        } else if (
          lower.includes('book') ||
          lower.includes('schedule') ||
          lower.includes('earliest') ||
          lower.includes('yes') ||
          lower.includes('tomorrow') ||
          lower.includes('thursday') ||
          lower.includes('friday')
        ) {
          const slotToBook = selectedSlot || availableSlots[0];
          const centerName = slotToBook
            ? typeof slotToBook.service_center === 'object'
              ? slotToBook.service_center.name
              : slotToBook.service_center
            : 'EV Care Mumbai Central';
          const time = slotToBook
            ? new Date(slotToBook.start_time).toLocaleString()
            : 'Tomorrow 10:00 AM';

          setSelectedTime(time);
          setCurrentStep('booking');

          try {
            const apiUrl =
              process.env.EXPO_PUBLIC_API_URL || 'http://10.79.239.149:8000';
            await fetch(`${apiUrl}/api/v1/serviceops/booking/auto-schedule`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                vehicle_id: 'VH005',
                customer_id: 'CUST005',
                failure_type: 'brake_fade',
                severity: 'critical',
                preferred_date: new Date().toISOString().split('T')[0],
                preferred_time_slot: 'morning',
              }),
            });
            onBookingComplete?.({ time, center: centerName });
          } catch (e) {
            console.error('Booking API failed:', e);
            onBookingComplete?.({ time, center: centerName });
          }

          addAIMessage(
            BRAKE_FADE_SCRIPTS.booking.ai.replace('{time}', time)
          );
        } else if (lower.includes('send') || lower.includes('details')) {
          setCurrentStep('sendDetails');
          addAIMessage(BRAKE_FADE_SCRIPTS.sendDetails.ai);
        } else if (lower.includes('expect')) {
          setCurrentStep('serviceExpectation');
          addAIMessage(BRAKE_FADE_SCRIPTS.serviceExpectation.ai);
        } else if (
          lower.includes('thanks') ||
          lower.includes('got it') ||
          lower.includes('no, that') ||
          lower.includes('good') ||
          lower.includes('bye') ||
          lower.includes('done')
        ) {
          setCurrentStep('farewell');
          addAIMessage(BRAKE_FADE_SCRIPTS.farewell.ai);
          setTimeout(() => onClose(), 4000);
        } else {
          setCurrentStep('recommendation');
          addAIMessage(BRAKE_FADE_SCRIPTS.recommendation.ai);
        }
      }, 800);
    },
    [addAIMessage, onClose, onBookingComplete, availableSlots, selectedSlot]
  );

  // Cleanup
  useEffect(() => {
    return () => {
      Speech.stop();
      speechWebViewRef.current?.stopListening();
    };
  }, []);

  if (!visible) return null;

  const currentScript = BRAKE_FADE_SCRIPTS[currentStep];

  return (
    <View style={styles.container}>
      <LinearGradient
        colors={['#0A0E1A', '#1A1F3C', '#0A0E1A']}
        style={styles.gradient}
      >
        {/* Hidden WebView for Speech Recognition */}
        <SpeechWebView
          ref={speechWebViewRef}
          onInterimResult={handleInterimResult}
          onFinalResult={handleFinalResult}
          onStatusChange={handleSpeechStatusChange}
          onError={handleSpeechError}
        />

        {/* Close Button */}
        <Pressable style={styles.closeButton} onPress={onClose}>
          <Text style={styles.closeText}>✕</Text>
        </Pressable>

        {/* Caller Info */}
        <View style={styles.callerSection}>
          <Animated.View
            style={[
              styles.avatarContainer,
              { transform: [{ scale: pulseAnim }] },
            ]}
          >
            <LinearGradient
              colors={['#00D9FF', '#7C3AED']}
              style={styles.avatar}
            >
              <Text style={styles.avatarText}>🤖</Text>
            </LinearGradient>

            {(isAISpeaking || isListening) &&
              [1, 2, 3].map((i) => (
                <Animated.View
                  key={i}
                  style={[
                    styles.pulseRing,
                    {
                      opacity: waveAnim.interpolate({
                        inputRange: [0, 1],
                        outputRange: [0.3 - i * 0.08, 0],
                      }),
                      transform: [
                        {
                          scale: waveAnim.interpolate({
                            inputRange: [0, 1],
                            outputRange: [1, 1 + i * 0.5],
                          }),
                        },
                      ],
                      borderColor: isListening ? '#FBBF24' : '#00D9FF',
                    },
                  ]}
                />
              ))}
          </Animated.View>

          <Text style={styles.callerName}>SentinEV Assistant</Text>
          <Text style={styles.callStatus}>
            {isAISpeaking
              ? '🔊 Speaking...'
              : isListening
              ? '👂 Listening...'
              : '🎙️ Tap mic or option to reply'}
          </Text>
        </View>

        <View style={{ flex: 1 }}>
          {/* Conversation History */}
          <ScrollView
            ref={scrollViewRef}
            style={styles.conversationContainer}
            contentContainerStyle={styles.conversationContent}
          >
            {conversationHistory.map((msg, idx) => (
              <View
                key={idx}
                style={[
                  styles.messageContainer,
                  msg.role === 'user' && styles.userMessageContainer,
                ]}
              >
                <Text
                  style={[
                    styles.messageText,
                    msg.role === 'user' && styles.userMessageText,
                  ]}
                >
                  {msg.text}
                </Text>
              </View>
            ))}

            {/* Live interim transcript */}
            {interimText ? (
              <View
                style={[
                  styles.messageContainer,
                  styles.userMessageContainer,
                  { opacity: 0.5, borderStyle: 'dashed' },
                ]}
              >
                <Text
                  style={[
                    styles.messageText,
                    styles.userMessageText,
                    { fontStyle: 'italic' },
                  ]}
                >
                  🎤 {interimText}...
                </Text>
              </View>
            ) : null}

            {/* Final recognized text (before match resolves) */}
            {finalText && !interimText ? (
              <View
                style={[
                  styles.messageContainer,
                  styles.userMessageContainer,
                  { opacity: 0.7 },
                ]}
              >
                <Text
                  style={[
                    styles.messageText,
                    styles.userMessageText,
                  ]}
                >
                  🎤 "{finalText}"
                </Text>
              </View>
            ) : null}

            {/* Match feedback */}
            {matchFeedback ? (
              <View style={styles.matchFeedbackContainer}>
                <Text style={styles.matchFeedbackText}>{matchFeedback}</Text>
              </View>
            ) : null}
          </ScrollView>

          {/* User Response Options */}
          {currentScript.userOptions.length > 0 && !isAISpeaking && (
            <View style={styles.optionsContainer}>
              {/* Microphone Button */}
              <Pressable
                style={[
                  styles.micButton,
                  isListening && styles.micButtonActive,
                ]}
                onPress={toggleListening}
              >
                <Text style={styles.micIcon}>
                  {isListening ? '⏹️' : '🎙️'}
                </Text>
                <Text style={styles.micLabel}>
                  {isListening ? 'Stop' : 'Speak'}
                </Text>
              </Pressable>

              <Text style={styles.orText}>— or tap to reply —</Text>

              {currentScript.userOptions.map((option, idx) => (
                <Pressable
                  key={idx}
                  style={styles.optionButton}
                  onPress={() => handleUserResponse(option)}
                >
                  <Text style={styles.optionText}>{option}</Text>
                </Pressable>
              ))}
            </View>
          )}

          {/* End Call */}
          <Pressable style={styles.endCallBtn} onPress={onClose}>
            <Text style={styles.endCallIcon}>📵</Text>
            <Text style={styles.endCallLabel}>End Call</Text>
          </Pressable>
        </View>
      </LinearGradient>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 999,
  },
  gradient: {
    flex: 1,
    paddingTop: 60,
    paddingBottom: Platform.OS === 'ios' ? 40 : 20,
    paddingHorizontal: 20,
  },
  closeButton: {
    position: 'absolute',
    top: 50,
    right: 20,
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.1)',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 10,
  },
  closeText: {
    color: '#FFF',
    fontSize: 18,
  },
  callerSection: {
    alignItems: 'center',
    marginBottom: 20,
  },
  avatarContainer: {
    width: 100,
    height: 100,
    marginBottom: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: { fontSize: 36 },
  pulseRing: {
    position: 'absolute',
    width: 80,
    height: 80,
    borderRadius: 40,
    borderWidth: 2,
    borderColor: '#00D9FF',
  },
  callerName: {
    fontSize: 20,
    fontWeight: '700',
    color: '#FFF',
    marginBottom: 4,
  },
  callStatus: {
    fontSize: 14,
    color: '#A0AEC0',
  },
  conversationContainer: {
    flex: 1,
    marginBottom: 12,
  },
  conversationContent: {
    paddingVertical: 8,
  },
  messageContainer: {
    backgroundColor: 'rgba(255,255,255,0.08)',
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
    maxWidth: '88%',
    alignSelf: 'flex-start',
  },
  userMessageContainer: {
    backgroundColor: '#7C3AED',
    alignSelf: 'flex-end',
  },
  messageText: {
    fontSize: 15,
    color: '#E2E8F0',
    lineHeight: 22,
  },
  userMessageText: {
    color: '#FFF',
  },
  matchFeedbackContainer: {
    alignSelf: 'center',
    backgroundColor: 'rgba(251, 191, 36, 0.15)',
    borderRadius: 12,
    paddingVertical: 6,
    paddingHorizontal: 14,
    marginBottom: 8,
  },
  matchFeedbackText: {
    color: '#FBBF24',
    fontSize: 13,
    fontWeight: '500',
  },
  optionsContainer: {
    marginBottom: 16,
    gap: 8,
    maxHeight: 260,
    alignItems: 'center',
  },
  micButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(251, 191, 36, 0.15)',
    paddingVertical: 14,
    paddingHorizontal: 28,
    borderRadius: 30,
    borderWidth: 1.5,
    borderColor: '#FBBF24',
    gap: 8,
    marginBottom: 4,
  },
  micButtonActive: {
    backgroundColor: 'rgba(251, 191, 36, 0.3)',
    borderColor: '#F59E0B',
  },
  micIcon: {
    fontSize: 20,
  },
  micLabel: {
    color: '#FBBF24',
    fontWeight: '700',
    fontSize: 16,
  },
  orText: {
    color: '#64748B',
    fontSize: 12,
    marginBottom: 2,
  },
  optionButton: {
    backgroundColor: 'rgba(0, 217, 255, 0.12)',
    borderWidth: 1,
    borderColor: 'rgba(0, 217, 255, 0.3)',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 18,
    alignItems: 'center',
    width: '100%',
  },
  optionText: {
    color: '#00D9FF',
    fontSize: 15,
    fontWeight: '600',
  },
  endCallBtn: {
    alignItems: 'center',
    backgroundColor: '#EF4444',
    width: 72,
    height: 72,
    borderRadius: 36,
    justifyContent: 'center',
    alignSelf: 'center',
    marginBottom: Platform.OS === 'ios' ? 20 : 10,
  },
  endCallIcon: { fontSize: 28 },
  endCallLabel: { fontSize: 11, color: '#FFF', marginTop: 4 },
});

export default BrakeFadeVoiceCall;