# Pitch Predictor Frontend

A React TypeScript application for predicting baseball pitches with a live game dashboard.

## Features

- **Pitch Predictor Panel**: Shows current at-bat context including pitcher, batter, pitch count, pitch history, and pitch predictions with probability levels
- **Live Game Dashboard**: Displays box scores and allows switching between different games
- **Minimalist Design**: Clean, modern UI with responsive layout
- **Real-time Updates**: Click on games to update the pitch predictor panel

## Quick Start with Docker

### Build the Docker Image

```bash
docker build -t pitch-predictor-frontend .
```

### Run the Application

```bash
# Run in foreground
docker run -p 3000:80 pitch-predictor-frontend

# Run in background (detached mode)
docker run -d -p 3000:80 pitch-predictor-frontend
```

### Access the Application

Open your browser and navigate to: **http://localhost:3000**

### Stop the Container

```bash
# List running containers
docker ps

# Stop the container
docker stop <container_id>
```

## Development Setup

### Prerequisites

- Node.js 20+
- npm

### Install Dependencies

```bash
npm install
```

### Run in Development Mode

```bash
npm start
```

The app will open at [http://localhost:3000](http://localhost:3000) with hot reload enabled.

### Build for Production

```bash
npm run build
```

This creates a `build` folder with optimized production files.

### Run Tests

```bash
npm test
```

## Project Structure

```
frontend/
├── src/
│   ├── App.tsx          # Main application component
│   ├── App.css          # Application styles
│   ├── index.tsx        # Application entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── Dockerfile           # Docker configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies and scripts
```

## Docker Configuration

The application uses a multi-stage Docker build:

1. **Build Stage**: Uses Node.js to install dependencies and build the React app
2. **Production Stage**: Uses nginx to serve the optimized build files

### Docker Commands

```bash
# Build image
docker build -t pitch-predictor-frontend .

# Run container
docker run -p 3000:80 pitch-predictor-frontend

# Run with custom port
docker run -p 8080:80 pitch-predictor-frontend

# Run in detached mode
docker run -d -p 3000:80 pitch-predictor-frontend

# View logs
docker logs <container_id>

# Stop container
docker stop <container_id>
```

## Technologies Used

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Create React App** - Build tooling
- **Docker** - Containerization
- **nginx** - Web server for production

## API Integration

The frontend is designed to integrate with a backend API for real-time pitch prediction data. Currently uses mock data for demonstration purposes.