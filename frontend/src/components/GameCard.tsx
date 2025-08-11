import React from 'react';
import { Game } from '../types/Game';
import { formatDateTime, getStatusColor } from '../utils/gameUtils';

interface GameCardProps {
  game: Game;
  isSelected: boolean;
  onSelect: () => void;
}

const GameCard: React.FC<GameCardProps> = ({ game, isSelected, onSelect }) => {
  return (
    <div
      className={`game-card ${isSelected ? 'selected' : ''}`}
      onClick={onSelect}
    >
      <div className={`status-badge ${getStatusColor(game.status)}`}>
        {game.status}
      </div>
      
      <div className="game-header">
        <div className="matchup">
          {game.summary}
        </div>
        <div className="game-time">
          {formatDateTime(game.game_datetime)}
        </div>
      </div>

      <div className="score-section">
        <div className="score-display">
          <span className="score">{game.away_score}</span>
          <span className="score-separator">-</span>
          <span className="score">{game.home_score}</span>
        </div>
        
        {game.current_inning > 0 && (
          <div className="inning-info">
            <span className="inning">
              {game.inning_state} {game.current_inning}
            </span>
          </div>
        )}
      </div>

      <div className="card-footer">
        <span className="click-hint">Click for details</span>
      </div>
    </div>
  );
};

export default GameCard;