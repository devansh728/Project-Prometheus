# SentinEV AdminOps - Executive Dashboard

Fleet-wide analytics and administration dashboard.

## Features

- 📊 **Executive Dashboard** - Fleet health, KPIs, failure trends
- 🔍 **RCA/CAPA Management** - Pattern detection, root cause analysis
- 🛡️ **UEBA Security** - Agent behavior monitoring, anomaly detection
- 📈 **Analytics Center** - Model accuracy, supplier scorecards

## Quick Start

```bash
npm install
npm run dev   # Opens on port 3002
```

## Project Structure

```
adminops/
├── src/
│   ├── components/     # Sidebar
│   ├── pages/
│   │   ├── Executive/  # Fleet health dashboard
│   │   └── Security/   # UEBA monitoring
│   ├── store/          # Zustand state
│   ├── types/          # TypeScript DTOs
│   ├── App.tsx
│   └── main.tsx
└── vite.config.ts
```

## Tech Stack

- Vite + React 18
- Recharts for visualizations
- Framer Motion for animations
- Zustand for state
- TypeScript
