// components/PlaysList.tsx
import React from 'react';
import { PredictedPitch } from '../types/Game';
import PredictionCard from './PredictionCard';
import PitchCard from './PitchCard';

interface PlaysListProps {
  plays: any[] | undefined;
  pitchCount: string | null;
  ballsCount: number;
  strikesCount: number;
  predictedPitch: PredictedPitch | null;
  gameStatus: string;
}

const PlaysList: React.FC<PlaysListProps> = ({
  plays,
  pitchCount,
  ballsCount,
  strikesCount,
  predictedPitch,
  gameStatus
}) => {
  if (!Array.isArray(plays)) {
    return <p className="no-data">No plays data available.</p>;
  }

  return (
    <>
      {plays.map((play: any, playIdx: number) => (
        <div key={playIdx} className="play-item">
          <div className="play-description">
            {play.dsc || "(No description)"}
            {playIdx === 0 && pitchCount !== null && (
              <> {" "} (Pitch count: {pitchCount}, Count: {ballsCount}-{strikesCount})</>
            )}
          </div>

          {Array.isArray(play.pitches) && play.pitches.length >= 0 && (
            <div className="pitches-container">
              <div className="pitches-header">Pitches:</div>
              <div className="pitches-grid">
                
                {/* Show prediction for first play if game is not final and the play is not over */}
                {playIdx === 0 && gameStatus !== "Final" && strikesCount < 3 && ballsCount < 4 && !play.dsc.toLowerCase().includes(' out ') && (
                  <PredictionCard predictedPitch={predictedPitch} />
                )}

                {/* Show actual pitches */}
                {play.pitches.map((pitch: any, pitchIdx: number) => (
                  <PitchCard
                    key={pitchIdx}
                    pitch={pitch}
                    pitchNumber={play.pitches.length - pitchIdx}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      ))}
    </>
  );
};

export default PlaysList;