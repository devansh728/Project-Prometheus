/**
 * Root Layout - Expo Router Configuration
 * Tab-based navigation with animated bottom bar
 */
import React, { useState, useEffect } from 'react';
import { View, StyleSheet, Text, ActivityIndicator } from 'react-native';
import { Tabs } from 'expo-router';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, {
  useAnimatedStyle,
  withSpring,
  useSharedValue,
} from 'react-native-reanimated';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import AsyncStorage from '@react-native-async-storage/async-storage';
import OnboardingScreen from '../src/screens/OnboardingScreen';
import { useStore } from '../src/store';

const theme = {
  colors: {
    background: '#0A0E1A',
    surface: '#141925',
    primary: '#00D9FF',
    textPrimary: '#FFFFFF',
    textMuted: '#64748B',
  },
};

interface TabIconProps {
  icon: string;
  label: string;
  focused: boolean;
}

const TabIcon: React.FC<TabIconProps> = ({ icon, label, focused }) => {
  const scale = useSharedValue(focused ? 1.1 : 1);

  React.useEffect(() => {
    scale.value = withSpring(focused ? 1.1 : 1);
  }, [focused]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return (
    <Animated.View style={[styles.tabItem, animatedStyle]}>
      <Text style={[styles.tabIcon, focused && styles.tabIconFocused]}>
        {icon}
      </Text>
      <Text style={[styles.tabLabel, focused && styles.tabLabelFocused]}>
        {label}
      </Text>
      {focused && <View style={styles.activeIndicator} />}
    </Animated.View>
  );
};

export default function RootLayout() {
  const [isLoading, setIsLoading] = useState(true);
  const { user, setUser, setSelectedVehicle } = useStore();

  useEffect(() => {
    checkRegistration();
  }, []);

  const checkRegistration = async () => {
    try {
      const userData = await AsyncStorage.getItem('sentinev_user');
      const vehicleData = await AsyncStorage.getItem('sentinev_vehicle');
      if (userData && vehicleData) {
        setUser(JSON.parse(userData));
        setSelectedVehicle(JSON.parse(vehicleData));
      }
    } catch (error) {
      console.error('Error checking registration:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleOnboardingComplete = async (user: any, vehicle: any) => {
    try {
      await AsyncStorage.setItem('sentinev_user', JSON.stringify(user));
      await AsyncStorage.setItem('sentinev_vehicle', JSON.stringify(vehicle));
      setUser(user);
      setSelectedVehicle(vehicle);
    } catch (error) {
      console.error('Error saving registration:', error);
    }
  };

  if (isLoading) {
    return (
      <View style={[styles.loadingContainer, { backgroundColor: theme.colors.background }]}>
        <ActivityIndicator size="large" color={theme.colors.primary} />
      </View>
    );
  }

  if (!user) {
    return (
      <GestureHandlerRootView style={{ flex: 1 }}>
        <SafeAreaProvider>
          <OnboardingScreen onComplete={handleOnboardingComplete} />
        </SafeAreaProvider>
      </GestureHandlerRootView>
    );
  }

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <SafeAreaProvider>
        <Tabs
          screenOptions={{
            headerShown: false,
            tabBarStyle: styles.tabBar,
            tabBarShowLabel: false,
          }}
        >
          <Tabs.Screen
            name="index"
            options={{
              tabBarIcon: ({ focused }) => (
                <TabIcon icon="🏠" label="Home" focused={focused} />
              ),
            }}
          />
          <Tabs.Screen
            name="vehicle"
            options={{
              tabBarIcon: ({ focused }) => (
                <TabIcon icon="🚗" label="Vehicle" focused={focused} />
              ),
            }}
          />
          <Tabs.Screen
            name="chat"
            options={{
              tabBarIcon: ({ focused }) => (
                <TabIcon icon="💬" label="Chat" focused={focused} />
              ),
            }}
          />
          <Tabs.Screen
            name="booking"
            options={{
              tabBarIcon: ({ focused }) => (
                <TabIcon icon="📅" label="Book" focused={focused} />
              ),
            }}
          />
        </Tabs>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  tabBar: {
    position: 'absolute',
    bottom: 20,
    left: 20,
    right: 20,
    backgroundColor: theme.colors.surface,
    borderRadius: 24,
    height: 70,
    borderTopWidth: 0,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.3,
    shadowRadius: 16,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.1)',
  },
  tabItem: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingTop: 10,
  },
  tabIcon: {
    fontSize: 22,
    marginBottom: 4,
  },
  tabIconFocused: {
    // Optional: add glow effect
  },
  tabLabel: {
    fontSize: 10,
    fontWeight: '500',
    color: theme.colors.textMuted,
  },
  tabLabelFocused: {
    color: theme.colors.primary,
  },
  activeIndicator: {
    position: 'absolute',
    bottom: -6,
    width: 4,
    height: 4,
    borderRadius: 2,
    backgroundColor: theme.colors.primary,
  },
});
