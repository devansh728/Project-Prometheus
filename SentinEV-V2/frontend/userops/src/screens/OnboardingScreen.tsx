/**
 * Onboarding Screen - User & Vehicle Registration
 * 3-step flow: User Profile → Vehicle Selection → Confirmation
 */
import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TextInput,
  Pressable,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  FadeInDown,
  FadeInUp,
  FadeOut,
  SlideInRight,
  SlideOutLeft,
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { theme } from '../theme';

// Valid credentials (hardcoded for demo)
const VALID_USERS = {
  'rahul': {
    name: 'Rahul Sharma',
    email: 'rahul.sharma@example.com',
    phone: '+919876543210',
  },
  'vikram': {
    name: 'Vikram Singh',
    email: 'vikram.singh@example.com',
    phone: '+919876543214',
  },
};

const VEHICLES = {
  'rahul': {
    id: 'VH001',
    vin: '1HGBH41JXMN109186',
    make: 'Tata',
    model: 'Nexon EV Max',
    year: 2024,
    mileage: 15420,
    healthScore: 92.5,
    category: 'normal',
    drivingProfile: 'eco',
  },
  'vikram': {
    id: 'VH005',
    vin: '5YFBURHE3HP581234',
    make: 'Kia',
    model: 'EV6',
    year: 2024,
    mileage: 12800,
    healthScore: 91.0,
    category: 'normal',
    drivingProfile: 'aggressive',
  },
};

type UserKey = 'rahul' | 'vikram';

interface OnboardingScreenProps {
  onComplete: (user: typeof VALID_USERS.rahul, vehicle: typeof VEHICLES.rahul) => void;
}

export default function OnboardingScreen({ onComplete }: OnboardingScreenProps) {
  const [step, setStep] = useState(1);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [errors, setErrors] = useState<{ name?: string; email?: string; phone?: string }>({});
  const [detectedUser, setDetectedUser] = useState<UserKey | null>(null);
  
  const shakeX = useSharedValue(0);
  
  const shakeStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: shakeX.value }],
  }));

  const triggerShake = () => {
    shakeX.value = withSequence(
      withTiming(-10, { duration: 50 }),
      withTiming(10, { duration: 50 }),
      withTiming(-10, { duration: 50 }),
      withTiming(10, { duration: 50 }),
      withTiming(0, { duration: 50 })
    );
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
  };

  const validateStep1 = () => {
    const newErrors: typeof errors = {};
    let matchedUser: UserKey | null = null;
    
    // Check which user matches
    for (const [key, user] of Object.entries(VALID_USERS) as [UserKey, typeof VALID_USERS.rahul][]) {
      if (
        name.trim() === user.name &&
        email.trim().toLowerCase() === user.email &&
        phone.trim().replace(/-/g, '') === user.phone
      ) {
        matchedUser = key;
        break;
      }
    }
    
    if (!matchedUser) {
      // Check individual fields for better error messages
      const nameMatch = Object.values(VALID_USERS).some(u => u.name === name.trim());
      const emailMatch = Object.values(VALID_USERS).some(u => u.email === email.trim().toLowerCase());
      const phoneMatch = Object.values(VALID_USERS).some(u => u.phone === phone.trim().replace(/-/g, ''));
      
      if (!nameMatch) newErrors.name = 'Try "Rahul Sharma" or "Vikram Singh"';
      if (!emailMatch) newErrors.email = 'Email doesn\'t match registered user';
      if (!phoneMatch) newErrors.phone = 'Phone doesn\'t match registered user';
    }
    
    setErrors(newErrors);
    
    if (matchedUser) {
      setDetectedUser(matchedUser);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setStep(2);
    } else {
      triggerShake();
    }
  };

  const confirmVehicle = () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setStep(3);
  };

  const finishOnboarding = () => {
    if (!detectedUser) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
    onComplete(VALID_USERS[detectedUser], isAddingNew ? SIMULATED_VEHICLE_DATA as any : VEHICLES[detectedUser]);
  };

  const currentUser = detectedUser ? VALID_USERS[detectedUser] : null;
  const currentVehicle = detectedUser ? VEHICLES[detectedUser] : null;

  const renderStep1 = () => (
    <Animated.View 
      entering={SlideInRight.springify()} 
      exiting={SlideOutLeft}
      style={styles.stepContainer}
    >
      <Text style={styles.stepTitle}>Welcome to SentinEV</Text>
      <Text style={styles.stepSubtitle}>Let's set up your profile</Text>
      
      <Animated.View style={[styles.inputContainer, shakeStyle]}>
        <Text style={styles.inputLabel}>Full Name</Text>
        <TextInput
          style={[styles.input, errors.name && styles.inputError]}
          placeholder="Enter your name"
          placeholderTextColor={theme.colors.textMuted}
          value={name}
          onChangeText={setName}
          autoCapitalize="words"
        />
        {errors.name && <Text style={styles.errorText}>{errors.name}</Text>}
      </Animated.View>

      <Animated.View style={[styles.inputContainer, shakeStyle]}>
        <Text style={styles.inputLabel}>Email Address</Text>
        <TextInput
          style={[styles.input, errors.email && styles.inputError]}
          placeholder="Enter your email"
          placeholderTextColor={theme.colors.textMuted}
          value={email}
          onChangeText={setEmail}
          keyboardType="email-address"
          autoCapitalize="none"
        />
        {errors.email && <Text style={styles.errorText}>{errors.email}</Text>}
      </Animated.View>

      <Animated.View style={[styles.inputContainer, shakeStyle]}>
        <Text style={styles.inputLabel}>Phone Number</Text>
        <TextInput
          style={[styles.input, errors.phone && styles.inputError]}
          placeholder="Enter your phone"
          placeholderTextColor={theme.colors.textMuted}
          value={phone}
          onChangeText={setPhone}
          keyboardType="phone-pad"
        />
        {errors.phone && <Text style={styles.errorText}>{errors.phone}</Text>}
      </Animated.View>

      <Pressable onPress={validateStep1}>
        <LinearGradient
          colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
          style={styles.button}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 0 }}
        >
          <Text style={styles.buttonText}>Continue</Text>
        </LinearGradient>
      </Pressable>
    </Animated.View>
  );

  // New state for "Add Vehicle" flow
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [newVehicleMake, setNewVehicleMake] = useState('');
  const [newVehicleModel, setNewVehicleModel] = useState('');
  const [newVehicleYear, setNewVehicleYear] = useState('');
  const [researchState, setResearchState] = useState<'idle' | 'uploading' | 'scanning' | 'analyzing' | 'complete'>('idle');
  const [researchProgress, setResearchProgress] = useState(0);
  const [researchLog, setResearchLog] = useState<string[]>([]);
  
  // Dummy data for new vehicle simulation
  const SIMULATED_VEHICLE_DATA = {
    id: 'VH-NEW-88',
    vin: '1N4AZ0CP5FCxxxxxx',
    make: newVehicleMake || 'Tesla',
    model: newVehicleModel || 'Model 3',
    year: parseInt(newVehicleYear) || 2024,
    mileage: 1200,
    healthScore: 98.5,
    category: 'new_entry',
    drivingProfile: 'balanced',
    baselineConfigs: ['Regen: Standard', 'Chill Mode: Off', 'Charge Limit: 80%'],
    detectedFaults: ['None (New Vehicle)', 'Minor Tire Pressure Variance'],
  };

  const simulateResearch = () => {
    setResearchState('uploading');
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

    // Sequence of simulated agent actions
    setTimeout(() => {
      setResearchState('scanning');
      setResearchLog(prev => [...prev, 'Reading manual PDF...']);
    }, 1500);

    setTimeout(() => {
      setResearchLog(prev => [...prev, 'Extracting technical specs...']);
      setResearchProgress(30);
    }, 2500);

    setTimeout(() => {
      setResearchState('analyzing');
      setResearchLog(prev => [...prev, 'Web Research Agent: Searching global databases...']);
      setResearchProgress(60);
    }, 4000);

    setTimeout(() => {
      setResearchLog(prev => [...prev, 'Comparing against fleet baselines...']);
      setResearchProgress(80);
    }, 5500);

    setTimeout(() => {
      setResearchState('complete');
      setResearchProgress(100);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    }, 7000);
  };

  const handleAddNewConfirm = () => {
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    // Use the simulated new vehicle data
    const finalVehicle = SIMULATED_VEHICLE_DATA;
    // We hack the filtered VEHICLES object or just pass this custom object
    // Since onComplete expects specific types, we might need to cast or adjust.
    // For demo, we'll just pass it as 'rahul's mock but with overridden data if possible,
    // or we modify the consuming component to handle it.
    // simpler: valid user + this new vehicle object.
    
    // We need to update Step 3 to show this vehicle's info
    // so we set a temporary "override" vehicle in state?
    // Let's use a ref or just rely on isAddingNew flag in step 3.
    setStep(3);
  };

  const renderResearchOverlay = () => {
    if (researchState === 'idle') return null;

    return (
      <Animated.View entering={FadeInDown} style={styles.researchContainer}>
        {researchState === 'uploading' && (
          <View style={styles.uploadingState}>
            <Text style={styles.uploadingText}>Uploading Manual...</Text>
            <View style={styles.progressBarBg}>
               <View style={[styles.progressBarFill, { width: '40%' }]} />
            </View>
          </View>
        )}

        {(researchState === 'scanning' || researchState === 'analyzing') && (
          <View style={styles.agentState}>
            <Text style={styles.agentTitle}>Web Research Agent Active</Text>
            <View style={styles.progressBarBg}>
               <Animated.View style={[styles.progressBarFill, { width: `${researchProgress}%` }]} />
            </View>
            <ScrollView style={styles.logContainer}>
              {researchLog.map((log, i) => (
                <Text key={i} style={styles.logText}>Result: {log}</Text>
              ))}
            </ScrollView>
          </View>
        )}

        {researchState === 'complete' && (
          <View style={styles.resultsState}>
            <Text style={styles.resultsTitle}>Analysis Complete</Text>
            <View style={styles.resultCard}>
              <Text style={styles.resultHeader}>Baseline Configs Found:</Text>
              {SIMULATED_VEHICLE_DATA.baselineConfigs.map((c, i) => (
                <Text key={i} style={styles.resultItem}>• {c}</Text>
              ))}
              <View style={styles.divider} />
              <Text style={styles.resultHeader}>Initial Health Check:</Text>
              {SIMULATED_VEHICLE_DATA.detectedFaults.map((f, i) => (
                 <Text key={i} style={styles.resultItem}>• {f}</Text>
              ))}
            </View>
             <Pressable onPress={handleAddNewConfirm}>
              <LinearGradient
                colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
                style={styles.button}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
              >
                <Text style={styles.buttonText}>Confirm New Vehicle</Text>
              </LinearGradient>
            </Pressable>
          </View>
        )}
      </Animated.View>
    );
  };

  const renderStep2 = () => (
    <Animated.View 
      entering={SlideInRight.springify()} 
      exiting={SlideOutLeft}
      style={styles.stepContainer}
    >
      <Text style={styles.stepTitle}>Your Vehicle</Text>
      
      {/* Toggle / Selection */}
      <View style={styles.toggleContainer}>
        <Pressable 
          style={[styles.toggleBtn, !isAddingNew && styles.toggleBtnActive]}
          onPress={() => setIsAddingNew(false)}
        >
          <Text style={[styles.toggleText, !isAddingNew && styles.toggleTextActive]}>Existing</Text>
        </Pressable>
        <Pressable 
          style={[styles.toggleBtn, isAddingNew && styles.toggleBtnActive]}
          onPress={() => setIsAddingNew(true)}
        >
          <Text style={[styles.toggleText, isAddingNew && styles.toggleTextActive]}>Add New</Text>
        </Pressable>
      </View>

      {!isAddingNew ? (
        // EXISTING FLOW
        <>
          <Text style={styles.stepSubtitle}>We found your registered vehicle</Text>
          <View style={styles.vehicleCard}>
             <LinearGradient
              colors={[theme.colors.cardGradientStart, theme.colors.cardGradientEnd]}
              style={styles.vehicleGradient}
            >
              <View style={styles.vehicleHeader}>
                <Text style={styles.vehicleEmoji}>🚗</Text>
                <View style={[
                  styles.vehicleBadge,
                  currentVehicle?.drivingProfile === 'aggressive' && styles.aggressiveBadge
                ]}>
                  <Text style={[
                    styles.badgeText,
                    currentVehicle?.drivingProfile === 'aggressive' && styles.aggressiveBadgeText
                  ]}>
                    {currentVehicle?.drivingProfile === 'aggressive' ? 'AGGRESSIVE' : 'ECO'}
                  </Text>
                </View>
              </View>
              
              <Text style={styles.vehicleName}>{currentVehicle?.make} {currentVehicle?.model}</Text>
              <Text style={styles.vehicleYear}>{currentVehicle?.year}</Text>
              
              <View style={styles.vehicleStats}>
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{currentVehicle?.healthScore}%</Text>
                  <Text style={styles.statLabel}>Health</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{((currentVehicle?.mileage || 0) / 1000).toFixed(1)}k</Text>
                  <Text style={styles.statLabel}>km</Text>
                </View>
                <View style={styles.statDivider} />
                <View style={styles.statItem}>
                  <Text style={styles.statValue}>{currentVehicle?.id}</Text>
                  <Text style={styles.statLabel}>ID</Text>
                </View>
              </View>
              
              <View style={styles.vinContainer}>
                <Text style={styles.vinLabel}>VIN</Text>
                <Text style={styles.vinValue}>{currentVehicle?.vin}</Text>
              </View>
            </LinearGradient>
          </View>

          <Pressable onPress={confirmVehicle}>
            <LinearGradient
              colors={theme.colors.gradientPrimary as unknown as readonly [string, string]}
              style={styles.button}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Text style={styles.buttonText}>Confirm Vehicle</Text>
            </LinearGradient>
          </Pressable>
        </>
      ) : (
        // ADD NEW FLOW
        <View>
          {researchState === 'idle' ? (
            <>
              <Text style={styles.stepSubtitle}>Enter details & upload manual</Text>
              
              <View style={styles.inputContainer}>
                 <Text style={styles.inputLabel}>Make</Text>
                 <TextInput 
                   style={styles.input} 
                   placeholder="e.g. Tesla" 
                   placeholderTextColor={theme.colors.textMuted}
                   value={newVehicleMake}
                   onChangeText={setNewVehicleMake}
                 />
              </View>
              <View style={styles.inputContainer}>
                 <Text style={styles.inputLabel}>Model</Text>
                 <TextInput 
                   style={styles.input} 
                   placeholder="e.g. Model 3" 
                   placeholderTextColor={theme.colors.textMuted}
                   value={newVehicleModel}
                   onChangeText={setNewVehicleModel}
                 />
              </View>
              <View style={styles.inputContainer}>
                 <Text style={styles.inputLabel}>Year</Text>
                 <TextInput 
                   style={styles.input} 
                   placeholder="2024" 
                   placeholderTextColor={theme.colors.textMuted}
                   value={newVehicleYear}
                   onChangeText={setNewVehicleYear}
                   keyboardType="numeric"
                 />
              </View>
              
              <Text style={styles.inputLabel}>Vehicle Manual (PDF)</Text>
              <Pressable style={styles.uploadBox} onPress={simulateResearch}>
                <Text style={styles.uploadEmoji}>📄</Text>
                <Text style={styles.uploadText}>Tap to Upload</Text>
              </Pressable>
            </>
          ) : (
             renderResearchOverlay()
          )}
        </View>
      )}
    </Animated.View>
  );

  const finalConfirmUser = detectedUser ? VALID_USERS[detectedUser] : null;
  // If adding new, we pretend it's the new vehicle, else the detected one
  const finalConfirmVehicle = isAddingNew ? SIMULATED_VEHICLE_DATA : (detectedUser ? VEHICLES[detectedUser] : null);

  const renderStep3 = () => (
    <Animated.View 
      entering={FadeInUp.springify()} 
      style={styles.stepContainer}
    >
      <Animated.View 
        entering={FadeInDown.delay(200).springify()}
        style={styles.successIcon}
      >
        <Text style={styles.successEmoji}>✅</Text>
      </Animated.View>
      
      <Animated.Text 
        entering={FadeInDown.delay(400).springify()}
        style={styles.successTitle}
      >
        You're All Set!
      </Animated.Text>
      
      <Animated.Text 
        entering={FadeInDown.delay(600).springify()}
        style={styles.successSubtitle}
      >
        Welcome, {finalConfirmUser?.name}!{'\n'}
        Your {finalConfirmVehicle?.make} {finalConfirmVehicle?.model} is connected.
      </Animated.Text>

      {isAddingNew && (
        <Animated.View entering={FadeInDown.delay(700)} style={styles.newBadge}>
           <Text style={styles.newBadgeText}>✨ Configured by AI Agent</Text>
        </Animated.View>
      )}

      <Animated.View entering={FadeInUp.delay(800).springify()}>
        <Pressable onPress={() => {
            if (!detectedUser) return;
             // Here we would ideally add the new vehicle to the global store/context
             // For now, we just proceed. The onComplete might need adjustment if it strictly requires the old type.
            onComplete(VALID_USERS[detectedUser], isAddingNew ? SIMULATED_VEHICLE_DATA as any : VEHICLES[detectedUser]);
        }}>
          <LinearGradient
            colors={theme.colors.gradientSuccess as unknown as readonly [string, string]}
            style={styles.button}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.buttonText}>Start Monitoring</Text>
          </LinearGradient>
        </Pressable>
      </Animated.View>
    </Animated.View>
  );


  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        style={styles.keyboardView}
      >
        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Progress Indicator */}
          <View style={styles.progressContainer}>
            {[1, 2, 3].map((s) => (
              <View key={s} style={styles.progressWrapper}>
                <View 
                  style={[
                    styles.progressDot,
                    s <= step && styles.progressDotActive,
                    s < step && styles.progressDotComplete,
                  ]}
                >
                  {s < step && <Text style={styles.checkmark}>✓</Text>}
                </View>
                {s < 3 && (
                  <View 
                    style={[
                      styles.progressLine,
                      s < step && styles.progressLineActive,
                    ]} 
                  />
                )}
              </View>
            ))}
          </View>

          {/* Step Content */}
          {step === 1 && renderStep1()}
          {step === 2 && renderStep2()}
          {step === 3 && renderStep3()}
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.background,
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingTop: 20,
  },
  progressContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 40,
  },
  progressWrapper: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  progressDot: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: theme.colors.surface,
    borderWidth: 2,
    borderColor: theme.colors.textMuted,
    justifyContent: 'center',
    alignItems: 'center',
  },
  progressDotActive: {
    borderColor: theme.colors.primary,
    backgroundColor: theme.colors.primaryDark,
  },
  progressDotComplete: {
    backgroundColor: theme.colors.success,
    borderColor: theme.colors.success,
  },
  checkmark: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  progressLine: {
    width: 40,
    height: 2,
    backgroundColor: theme.colors.textMuted,
    marginHorizontal: 4,
  },
  progressLineActive: {
    backgroundColor: theme.colors.success,
  },
  stepContainer: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 28,
    fontWeight: '700',
    color: theme.colors.textPrimary,
    marginBottom: 8,
  },
  stepSubtitle: {
    fontSize: 16,
    color: theme.colors.textMuted,
    marginBottom: 32,
  },
  // ... existing styles ...
  
  // New Styles for Add Vehicle Flow
  toggleContainer: {
    flexDirection: 'row',
    backgroundColor: theme.colors.surface,
    borderRadius: 12,
    padding: 4,
    marginBottom: 24,
  },
  toggleBtn: {
    flex: 1,
    paddingVertical: 10,
    alignItems: 'center',
    borderRadius: 10,
  },
  toggleBtnActive: {
    backgroundColor: theme.colors.primary, // Using primary color for active state visibility
  },
  toggleText: {
    color: theme.colors.textMuted,
    fontWeight: '600',
  },
  toggleTextActive: {
    color: '#fff',
    fontWeight: '700',
  },
  uploadBox: {
    borderWidth: 2,
    borderColor: theme.colors.glassBorder,
    borderStyle: 'dashed',
    borderRadius: 16,
    height: 120,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 8,
    marginBottom: 20,
    backgroundColor: 'rgba(255,255,255,0.02)',
  },
  uploadEmoji: {
    fontSize: 32,
    marginBottom: 8,
  },
  uploadText: {
    color: theme.colors.primary,
    fontWeight: '600',
  },
  researchContainer: {
    marginVertical: 20,
    backgroundColor: theme.colors.surface,
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  uploadingState: {
    alignItems: 'center',
  },
  uploadingText: {
    color: theme.colors.textPrimary,
    marginBottom: 12,
    fontWeight: '600',
  },
  progressBarBg: {
    width: '100%',
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: theme.colors.primary,
    borderRadius: 3,
  },
  agentState: {
    width: '100%',
  },
  agentTitle: {
    color: theme.colors.textPrimary,
    fontWeight: '700',
    marginBottom: 12,
    fontSize: 16,
  },
  logContainer: {
    marginTop: 16,
    maxHeight: 120,
  },
  logText: {
    color: theme.colors.primary,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
    fontSize: 12,
    marginBottom: 4,
  },
  resultsState: {
    width: '100%',
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: theme.colors.success,
    marginBottom: 16,
    textAlign: 'center',
  },
  resultCard: {
    backgroundColor: 'rgba(0,0,0,0.2)',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
  },
  resultHeader: {
    color: theme.colors.textMuted,
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
    textTransform: 'uppercase',
  },
  resultItem: {
    color: theme.colors.textPrimary,
    fontSize: 14,
    marginBottom: 4,
    paddingLeft: 8,
  },
  divider: {
    height: 1,
    backgroundColor: theme.colors.glassBorder,
    marginVertical: 12,
  },
  newBadge: {
    backgroundColor: theme.colors.primary,
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    alignSelf: 'center',
    marginBottom: 20,
  },
  newBadgeText: {
    color: '#fff',
    fontWeight: '700',
    fontSize: 14,
  },
  inputContainer: {
    marginBottom: 20,
  },
  inputLabel: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.textSecondary,
    marginBottom: 8,
  },
  input: {
    backgroundColor: theme.colors.surface,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    color: theme.colors.textPrimary,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
  },
  inputError: {
    borderColor: theme.colors.danger,
  },
  errorText: {
    color: theme.colors.danger,
    fontSize: 12,
    marginTop: 4,
  },
  button: {
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 24,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
  vehicleCard: {
    borderRadius: 20,
    overflow: 'hidden',
    marginBottom: 20,
  },
  vehicleGradient: {
    padding: 24,
    borderWidth: 1,
    borderColor: theme.colors.glassBorder,
    borderRadius: 20,
  },
  vehicleHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  vehicleEmoji: {
    fontSize: 48,
  },
  vehicleBadge: {
    backgroundColor: theme.colors.successSoft,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  aggressiveBadge: {
    backgroundColor: theme.colors.warningSoft,
  },
  badgeText: {
    color: theme.colors.success,
    fontSize: 12,
    fontWeight: '700',
  },
  aggressiveBadgeText: {
    color: theme.colors.warning,
  },
  vehicleName: {
    fontSize: 24,
    fontWeight: '700',
    color: theme.colors.textPrimary,
  },
  vehicleYear: {
    fontSize: 16,
    color: theme.colors.textMuted,
    marginBottom: 20,
  },
  vehicleStats: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    paddingVertical: 16,
    borderTopWidth: 1,
    borderBottomWidth: 1,
    borderColor: theme.colors.glassBorder,
    marginBottom: 16,
  },
  statItem: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 20,
    fontWeight: '700',
    color: theme.colors.primary,
  },
  statLabel: {
    fontSize: 12,
    color: theme.colors.textMuted,
    marginTop: 4,
  },
  statDivider: {
    width: 1,
    backgroundColor: theme.colors.glassBorder,
  },
  vinContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  vinLabel: {
    fontSize: 12,
    color: theme.colors.textMuted,
    marginRight: 8,
  },
  vinValue: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  successIcon: {
    alignItems: 'center',
    marginBottom: 24,
  },
  successEmoji: {
    fontSize: 80,
  },
  successTitle: {
    fontSize: 32,
    fontWeight: '700',
    color: theme.colors.textPrimary,
    textAlign: 'center',
    marginBottom: 12,
  },
  successSubtitle: {
    fontSize: 16,
    color: theme.colors.textMuted,
    textAlign: 'center',
    lineHeight: 24,
    marginBottom: 40,
  },
});
