import React from 'react';
import { Game, PredictedPitch, CurrentInningData } from '../types/Game';
import { formatDateTime, getStatusColor } from '../utils/gameUtils';
import BaseIndicator from './BaseIndicator';
import PlaysList from './PlaysList';

interface GameDetailsPanelProps {
  isVisible: boolean;
  selectedGame: Game | undefined;
  currentInningData: CurrentInningData | null;
  predictedPitch: PredictedPitch | null;
  isLoading: boolean;
  onClose: () => void;
}

const GameDetailsPanel: React.FC<GameDetailsPanelProps> = ({
  isVisible,
  selectedGame,
  currentInningData,
  predictedPitch,
  isLoading,
  onClose
}) => {
  if (!selectedGame) return null;

  return (
    <div className={`details-overlay ${isVisible ? 'active' : ''}`}>
      <div className="details-panel">
        <div className="details-header">
          <h2>Game Details</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>
        
        <div className="details-content">
          <div className="game-summary-section">
            <h3>{selectedGame.summary}</h3>
            <div className="score-large">
              {selectedGame.away_score} - {selectedGame.home_score}
            </div>
            <div className="game-meta">
              <span className={`status ${getStatusColor(selectedGame.status)}`}>
                {selectedGame.status}
              </span>
              <span className="datetime">
                {formatDateTime(selectedGame.game_datetime)}
              </span>
              {selectedGame.current_inning > 0 && (
                <span className="inning">
                  {selectedGame.inning_state} {selectedGame.current_inning}
                </span>
              )}
            </div>
          </div>

          <div className={`play-by-play-section${isLoading ? ' fading' : ''}`}>
            <h4>Play-by-Play: Inning {selectedGame.current_inning} ({selectedGame.inning_state})</h4>
            {currentInningData?.playByPlay ? (
              <div className="inning-stats">
                <div className="stat-grid">
                  <div className="stat-item">
                    <span className="stat-label">Runs</span>
                    <span className="stat-value">{currentInningData.playByPlay.runs}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Hits</span>
                    <span className="stat-value">{currentInningData.playByPlay.hits}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Errors</span>
                    <span className="stat-value">{currentInningData.playByPlay.errors}</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Outs</span>
                    <span className="stat-value">{currentInningData.outsCount}</span>
                  </div>
                </div>

                <div className="plays-list">
                  <h5>Recent Plays</h5>
                  <BaseIndicator bases={currentInningData.bases} />
                  <PlaysList
                    plays={currentInningData.playByPlay.plays}
                    pitchCount={currentInningData.pitchCount}
                    ballsCount={currentInningData.ballsCount}
                    strikesCount={currentInningData.strikesCount}
                    predictedPitch={predictedPitch}
                    gameStatus={selectedGame.status}
                  />
                </div>
              </div>
            ) : (
              <p className="no-data">No play-by-play data available for this inning.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GameDetailsPanel;