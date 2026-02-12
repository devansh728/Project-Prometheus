# SentinEV ServiceOps - Admin Dashboard

Service center operations dashboard for technicians and managers.

## Features

- 📊 **Command Center** - Real-time KPI metrics and job pipeline
- 📋 **Kanban Board** - Visual job workflow management
- 👥 **Technician Status** - Workload and utilization tracking
- 📦 **Inventory Alerts** - Low stock notifications
- 🎨 **Premium Dark Theme** - Glassmorphism and animations

## Quick Start

```bash
# Install dependencies
npm install

# Start development server (port 3001)
npm run dev

# Build for production
npm run build
```

## Project Structure

```
serviceops/
├── src/
│   ├── components/
│   │   ├── layout/         # Sidebar, Header
│   │   ├── cards/          # MetricCard, JobCard
│   │   └── kanban/         # KanbanBoard
│   ├── pages/
│   │   └── Dashboard/      # Command Center
│   ├── services/           # API client
│   ├── store/              # Zustand state
│   ├── types/              # TypeScript DTOs
│   ├── App.tsx             # Main layout
│   ├── main.tsx            # Entry point
│   └── index.css           # Global styles
└── vite.config.ts          # Vite configuration
```

## API Integration

The dashboard integrates with the SentinEV ServiceOps backend:

- `GET /api/v1/serviceops/jobs` - List service jobs
- `POST /api/v1/serviceops/jobs/{id}/transition` - Update job state
- `GET /api/v1/serviceops/workload/{center}` - Technician workload
- `GET /api/v1/serviceops/inventory/{center}` - Parts inventory

## Tech Stack

- **Vite** + React 18
- **Framer Motion** for animations
- **Recharts** for data visualization
- **Zustand** for state management
- **Lucide React** for icons
- **TypeScript** for type safety
