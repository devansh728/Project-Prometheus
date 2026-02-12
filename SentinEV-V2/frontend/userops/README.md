# SentinEV UserOps - Mobile Application

Premium EV Predictive Maintenance mobile app built with React Native (Expo).

## Features

- 🎨 **Premium Dark Theme** - Glassmorphism, gradients, and glow effects
- ✨ **Animated Components** - Smooth transitions with Reanimated 3
- 📊 **Real-time Telemetry** - Animated gauges for vehicle health
- 💬 **AI Chat** - Contextual assistant with typing indicators
- 📱 **Incoming Call UI** - Full-screen overlay with ring animations
- 📅 **Service Booking** - Slot selection with animated confirmation

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm start

# Run on iOS
npm run ios

# Run on Android
npm run android
```

## Environment Variables

Create a `.env` file:

```
EXPO_PUBLIC_API_URL=http://localhost:8000/api/v1
EXPO_PUBLIC_WS_URL=ws://localhost:8000/api/v1
```

## Project Structure

```
userops/
├── app/                    # Expo Router pages
│   ├── _layout.tsx        # Tab navigation
│   ├── index.tsx          # Dashboard
│   ├── vehicle.tsx        # Vehicle details
│   ├── chat.tsx           # AI chat
│   └── booking.tsx        # Service booking
├── src/
│   ├── components/        # Reusable components
│   │   ├── cards/         # HealthCard, etc.
│   │   ├── buttons/       # QuickAction, etc.
│   │   └── overlays/      # IncomingCall, etc.
│   ├── screens/           # Screen components
│   ├── services/          # API & WebSocket
│   ├── store/             # Zustand state
│   ├── theme/             # Theme tokens
│   └── types/             # TypeScript DTOs
└── assets/                # Images & fonts
```

## API Integration

The app integrates with the SentinEV backend via:

- **REST API** - Vehicle analysis, scheduling, RAG queries
- **WebSocket** - Real-time telemetry streaming, agent workflow events

## Tech Stack

- **Expo SDK 50** with Expo Router
- **React Native Reanimated 3** for animations
- **Zustand** for state management
- **Axios** for HTTP requests
- **TypeScript** for type safety
