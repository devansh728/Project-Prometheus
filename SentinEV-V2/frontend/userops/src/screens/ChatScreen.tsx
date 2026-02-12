/**
 * Chat Screen - AI Assistant Interface
 * Rich messaging with typing indicators and animated messages
 */
import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  FlatList,
  Pressable,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import Animated, {
  FadeInRight,
  FadeInLeft,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';

import { theme } from '../theme';
import { useStore } from '../store';
import { ChatMessage, Severity } from '../types/api';

export default function ChatScreen() {
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  
  const { chatMessages, addChatMessage, selectedVehicle, vehicleHealth } = useStore();

  // Typing indicator animation
  const dot1 = useSharedValue(0);
  const dot2 = useSharedValue(0);
  const dot3 = useSharedValue(0);

  useEffect(() => {
    if (isTyping) {
      dot1.value = withRepeat(
        withSequence(
          withTiming(-5, { duration: 300 }),
          withTiming(0, { duration: 300 })
        ),
        -1
      );
      dot2.value = withRepeat(
        withSequence(
          withTiming(0, { duration: 150 }),
          withTiming(-5, { duration: 300 }),
          withTiming(0, { duration: 300 })
        ),
        -1
      );
      dot3.value = withRepeat(
        withSequence(
          withTiming(0, { duration: 300 }),
          withTiming(-5, { duration: 300 }),
          withTiming(0, { duration: 300 })
        ),
        -1
      );
    }
  }, [isTyping]);

  const dot1Style = useAnimatedStyle(() => ({
    transform: [{ translateY: dot1.value }],
  }));
  const dot2Style = useAnimatedStyle(() => ({
    transform: [{ translateY: dot2.value }],
  }));
  const dot3Style = useAnimatedStyle(() => ({
    transform: [{ translateY: dot3.value }],
  }));

  const handleSend = async () => {
    if (!inputText.trim()) return;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText,
      timestamp: new Date().toISOString(),
    };
    addChatMessage(userMessage);
    const savedInput = inputText;
    setInputText('');

    // Simulate AI response
    setIsTyping(true);
    
    setTimeout(() => {
      setIsTyping(false);
      
      // Generate contextual response for healthy vehicle demo
      let responseContent = '';
      const lowerInput = savedInput.toLowerCase();
      
      // Health queries - main demo responses
      if (lowerInput.includes('health') || lowerInput.includes('how is my') || lowerInput.includes('vehicle status')) {
        responseContent = `✨ **Excellent News!**\n\nYour ${selectedVehicle?.make || 'Tata'} ${selectedVehicle?.model || 'Nexon EV Max'} is in outstanding condition!\n\n📊 **Health Score: 94/100**\n\n🔋 Battery: 98% capacity, optimal\n🛞 Brakes: 82% life remaining\n🌡️ Cooling System: Normal (85°C)\n⚡ Motor: Operating perfectly\n\n🛡️ No anomalies detected. All systems are operating within optimal parameters.\n\nYour eco-driving profile has contributed to excellent vehicle longevity! Keep up the great driving habits! 🌿`;
      }
      // Anomaly/issue queries
      else if (lowerInput.includes('issue') || lowerInput.includes('problem') || lowerInput.includes('anomal') || lowerInput.includes('wrong')) {
        responseContent = `🟢 **No Issues Detected**\n\nI've performed a comprehensive diagnostic scan of your vehicle:\n\n✅ Battery System: Healthy\n✅ Brake System: Healthy  \n✅ Motor & Drivetrain: Healthy\n✅ Cooling System: Healthy\n✅ Electronics: Healthy\n\n📊 Anomaly Score: 0.03 (Very Low)\n⚠️ Failure Probability: 0.5%\n\nYour vehicle is operating exactly as expected. No maintenance intervention is required at this time.`;
      }
      // Service/maintenance queries
      else if (lowerInput.includes('service') || lowerInput.includes('maintenance') || lowerInput.includes('next')) {
        responseContent = `📅 **Service Schedule**\n\nBased on your current vehicle condition and driving patterns:\n\n🔧 Next Routine Service: In 45 days\n📍 Recommended Center: EV Care Mumbai Central (⭐ 4.8)\n\n**No Urgent Service Required**\n\nYour proactive maintenance approach through SentinEV's monitoring has helped avoid unnecessary service visits. The AI system deliberately avoids creating false alarms for healthy vehicles like yours.\n\n💡 Tip: Continue your eco-driving habits to maximize battery life!`;
      }
      // Driving score queries  
      else if (lowerInput.includes('score') || lowerInput.includes('driving') || lowerInput.includes('points')) {
        responseContent = `🏆 **Safe Driving Score: 94**\n\nCongratulations! You're in the top 5% of eco-drivers!\n\n📈 **Your Achievements:**\n✅ 2,450 Total Points\n🔥 7-Day Driving Streak\n🌿 Eco Champion Badge\n🛡️ Safe Driver Badge\n⚡ EV Pioneer Badge\n\n**Recent Activity:**\n+15 points (This session)\n+10 points (Smooth braking)\n+5 points (Optimal speed)\n\nKeep driving safely to unlock more rewards! 🎯`;
      }
      // Booking/schedule queries
      else if (lowerInput.includes('book') || lowerInput.includes('schedule') || lowerInput.includes('appointment')) {
        responseContent = `📅 **Service Booking**\n\nSince your vehicle is in excellent health, no immediate service is needed. However, if you'd like to schedule a routine check-up:\n\n🏢 **EV Care Mumbai Central**\n📍 2.3 km away\n⭐ 4.8 rating\n🕐 Next available: Tomorrow, 10:00 AM\n\n🏢 **GreenDrive Pune**\n📍 8.5 km away  \n⭐ 4.5 rating\n🕐 Next available: Today, 3:00 PM\n\nWould you like me to book a slot?`;
      }
      // Battery queries
      else if (lowerInput.includes('battery') || lowerInput.includes('charge') || lowerInput.includes('range')) {
        responseContent = `🔋 **Battery Status: Excellent**\n\nYour EV battery is performing optimally:\n\n📊 Degradation: Minimal (2.5% from new)\n⚡ Estimated Range: 340 km\n🌡️ Temperature: Optimal\n🔌 Charging Health: 98%\n\n💡 Your eco-driving profile and consistent charging habits have preserved excellent battery health!\n\nRemaining Useful Life: 2,400+ hours`;
      }
      // Brake queries
      else if (lowerInput.includes('brake')) {
        responseContent = `🛞 **Brake System: Healthy**\n\n✅ Brake Pad Life: 82% remaining\n✅ Brake Fluid: Optimal level\n✅ Regenerative Braking: Functioning perfectly\n✅ Pressure: 48 PSI (Normal range)\n\nNo brake-related issues detected. The regenerative braking system is helping preserve brake pad life effectively.\n\n📅 Estimated replacement: Not needed for 18+ months`;
      }
      // Default response
      else {
        responseContent = `I'm here to help with anything about your ${selectedVehicle?.make || 'Tata'} ${selectedVehicle?.model || 'Nexon EV Max'}!\n\n**Quick Suggestions:**\n• "How is my vehicle health today?"\n• "Any issues detected?"\n• "Show my driving score"\n• "When is my next service?"\n• "Battery status"\n\nYour vehicle is currently in excellent condition with no anomalies detected. 🟢`;
      }

      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: responseContent,
        timestamp: new Date().toISOString(),
      };
      addChatMessage(aiMessage);
    }, 1500);
  };

  const renderMessage = ({ item, index }: { item: ChatMessage; index: number }) => {
    const isUser = item.role === 'user';
    const Animation = isUser ? FadeInRight : FadeInLeft;

    return (
      <Animated.View
        entering={Animation.delay(index * 50).springify()}
        style={[
          styles.messageContainer,
          isUser ? styles.userMessageContainer : styles.aiMessageContainer,
        ]}
      >
        {!isUser && (
          <View style={styles.avatarContainer}>
            <LinearGradient
              colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
              style={styles.avatar}
            >
              <Text style={styles.avatarText}>🤖</Text>
            </LinearGradient>
          </View>
        )}
        
        <View
          style={[
            styles.messageBubble,
            isUser ? styles.userBubble : styles.aiBubble,
          ]}
        >
          <Text style={[styles.messageText, isUser && styles.userMessageText]}>
            {item.content}
          </Text>
          <Text style={[styles.timestamp, isUser && styles.userTimestamp]}>
            {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </Text>
        </View>
      </Animated.View>
    );
  };

  return (
    <View style={styles.container}>
      <SafeAreaView style={styles.safeArea} edges={['top', 'left', 'right']}>
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerContent}>
            <LinearGradient
              colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
              style={styles.headerAvatar}
            >
              <Text style={styles.headerAvatarText}>🤖</Text>
            </LinearGradient>
            <View style={styles.headerText}>
              <Text style={styles.headerTitle}>SentinEV Assistant</Text>
              <View style={styles.onlineStatus}>
                <View style={styles.onlineDot} />
                <Text style={styles.onlineText}>Online</Text>
              </View>
            </View>
          </View>
        </View>

        <KeyboardAvoidingView 
          style={styles.chatContainer}
          behavior={Platform.OS === 'ios' ? 'padding' : undefined}
          keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
        >
          {/* Messages */}
          <FlatList
            ref={flatListRef}
            data={chatMessages}
            renderItem={renderMessage}
            keyExtractor={(item) => item.id}
            contentContainerStyle={styles.messagesList}
            onContentSizeChange={() => flatListRef.current?.scrollToEnd()}
            showsVerticalScrollIndicator={false}
          />

          {/* Typing Indicator */}
          {isTyping && (
            <View style={styles.typingContainer}>
              <View style={styles.typingBubble}>
                <Animated.View style={[styles.typingDot, dot1Style]} />
                <Animated.View style={[styles.typingDot, dot2Style]} />
                <Animated.View style={[styles.typingDot, dot3Style]} />
              </View>
            </View>
          )}

          {/* Input */}
          <View style={styles.inputContainer}>
            <View style={styles.inputWrapper}>
              <TextInput
                style={styles.input}
                placeholder="Ask about your vehicle..."
                placeholderTextColor={theme.colors.textMuted}
                value={inputText}
                onChangeText={setInputText}
                multiline
                maxLength={500}
              />
              <Pressable
                style={[styles.sendButton, !inputText.trim() && styles.sendButtonDisabled]}
                onPress={handleSend}
                disabled={!inputText.trim()}
              >
                <LinearGradient
                  colors={inputText.trim() 
                    ? theme.colors.gradientPrimary as unknown as readonly [string, string]
                    : [theme.colors.textMuted, theme.colors.textMuted]
                  }
                  style={styles.sendButtonGradient}
                >
                  <Text style={styles.sendButtonText}>↑</Text>
                </LinearGradient>
              </Pressable>
            </View>
          </View>
        </KeyboardAvoidingView>
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  safeArea: {
    flex: 1,
  },
  header: {
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.glassBorder,
  },
  headerContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  headerAvatar: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
  },
  headerAvatarText: {
    fontSize: 20,
  },
  headerText: {
    marginLeft: theme.spacing.md,
  },
  headerTitle: {
    ...theme.typography.h3,
    color: theme.colors.textPrimary,
  },
  onlineStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 2,
  },
  onlineDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.success,
    marginRight: 6,
  },
  onlineText: {
    ...theme.typography.caption,
    color: theme.colors.success,
  },
  chatContainer: {
    flex: 1,
  },
  messagesList: {
    padding: theme.spacing.md,
    paddingBottom: 120, // Extra space for input + tab bar
  },
  messageContainer: {
    marginBottom: theme.spacing.md,
    flexDirection: 'row',
  },
  userMessageContainer: {
    justifyContent: 'flex-end',
  },
  aiMessageContainer: {
    justifyContent: 'flex-start',
  },
  avatarContainer: {
    marginRight: theme.spacing.sm,
  },
  avatar: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    fontSize: 16,
  },
  messageBubble: {
    maxWidth: '75%',
    padding: theme.spacing.md,
    borderRadius: theme.borderRadius.lg,
  },
  userBubble: {
    backgroundColor: theme.colors.primary,
    borderBottomRightRadius: 4,
  },
  aiBubble: {
    backgroundColor: theme.colors.surfaceElevated,
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  messageText: {
    ...theme.typography.body,
    color: theme.colors.textPrimary,
  },
  userMessageText: {
    color: '#0A0E1A',
  },
  timestamp: {
    ...theme.typography.caption,
    color: theme.colors.textMuted,
    marginTop: theme.spacing.xs,
    textAlign: 'right',
  },
  userTimestamp: {
    color: 'rgba(0,0,0,0.5)',
  },
  typingContainer: {
    paddingHorizontal: theme.spacing.md,
    paddingBottom: theme.spacing.sm,
  },
  typingBubble: {
    flexDirection: 'row',
    backgroundColor: theme.colors.surfaceElevated,
    padding: theme.spacing.md,
    borderRadius: theme.borderRadius.lg,
    alignSelf: 'flex-start',
    marginLeft: 40,
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: theme.colors.textMuted,
    marginHorizontal: 3,
  },
  inputContainer: {
    padding: theme.spacing.md,
    paddingBottom: theme.spacing.md + 90, // Tab bar (70) + margin (20)
    borderTopWidth: 1,
    borderTopColor: theme.colors.glassBorder,
    backgroundColor: theme.colors.background, // Ensure opaque background
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: theme.colors.surfaceElevated,
    borderRadius: theme.borderRadius.xl,
    paddingHorizontal: theme.spacing.md,
    paddingVertical: theme.spacing.sm,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  input: {
    flex: 1,
    ...theme.typography.body,
    color: theme.colors.textPrimary,
    maxHeight: 100,
    paddingVertical: theme.spacing.xs,
  },
  sendButton: {
    marginLeft: theme.spacing.sm,
  },
  sendButtonDisabled: {
    opacity: 0.5,
  },
  sendButtonGradient: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  sendButtonText: {
    fontSize: 20,
    fontWeight: '700',
    color: '#0A0E1A',
  },
});
