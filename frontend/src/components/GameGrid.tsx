import React from 'react';
import { Game } from '../types/Game';
import GameCard from './GameCard';

interface GameGridProps {
  games: Game[];
  selectedGame: Game | undefined;
  onGameSelect: (game: Game) => void;
}

const GameGrid: React.FC<GameGridProps> = ({ games, selectedGame, onGameSelect }) => {
  return (
    <div className="games-grid">
      {games.map((game) => (
        <GameCard
          key={game.game_id}
          game={game}
          isSelected={selectedGame?.game_id === game.game_id}
          onSelect={() => onGameSelect(game)}
        />
      ))}
    </div>
  );
};

export default GameGrid;