import React, { useState, useEffect } from 'react';
import './App.css';
import { Game } from './types/Game';
import { useGameData } from './hooks/useGameData';
import { usePlayByPlay } from './hooks/usePlayByPlay';
import { usePrediction } from './hooks/usePrediction';
import GameGrid from './components/GameGrid';
import GameDetailsPanel from './components/GameDetailsPanel';
import LoadingScreen from './components/LoadingScreen';
import Header from './components/Header';

function App() {
  const [selectedGame, setSelectedGame] = useState<Game | undefined>(undefined);
  const [showDetails, setShowDetails] = useState(false);

  // Custom hooks to manage different data concerns
  const { games, isLoading } = useGameData();
  const { playByPlay, currentInningData, isPlayByPlayLoading } = usePlayByPlay( selectedGame?.espn_id, showDetails);
  const { predictedPitch } = usePrediction(currentInningData, selectedGame);

  const handleGameSelect = (game: Game) => {
    if (selectedGame?.game_id === game.game_id && showDetails) {
      setShowDetails(false);
    } else {
      setSelectedGame(game);
      setShowDetails(true);
    }
  };

  const closeDetails = () => {
    setShowDetails(false);
  };

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <div className="app">
      <Header />
      
      <main className="main">
        <GameGrid 
          games={games}
          selectedGame={selectedGame}
          onGameSelect={handleGameSelect}
        />

        <GameDetailsPanel
          isVisible={showDetails}
          selectedGame={selectedGame}
          currentInningData={currentInningData}
          predictedPitch={predictedPitch}
          isLoading={isPlayByPlayLoading}
          onClose={closeDetails}
        />
      </main>
    </div>
  );
}

export default App;