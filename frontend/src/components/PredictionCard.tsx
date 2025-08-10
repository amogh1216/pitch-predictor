// components/PredictionCard.tsx
import React from 'react';
import { PredictedPitch } from '../types/Game';

interface PredictionCardProps {
  predictedPitch: PredictedPitch | null;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ predictedPitch }) => {
  if (!predictedPitch) {
    return (
      <div className="pitch-card predicted-pitch">
        <div className="pitch-number">Predicted</div>
        <div className="pitch-speed predicted-highest">
          Waiting on data...
        </div>
      </div>
    );
  }

  const predictions = [
    { type: 'FAST', value: predictedPitch.FAST || 0 },
    { type: 'OFF', value: predictedPitch.OFF || 0 },
    { type: 'BREAK', value: predictedPitch.BREAK || 0 },
    { type: 'OTH', value: predictedPitch.OTH || 0 }
  ];

  // Find the highest probability
  const highest = predictions.reduce((max, current) => 
    current.value > max.value ? current : max
  );

  // Sort others by value descending, excluding the highest
  const others = predictions
    .filter(p => p.type !== highest.type)
    .sort((a, b) => b.value - a.value);

  return (
    <div className="pitch-card predicted-pitch">
      <div className="pitch-number">Predicted</div>
      <div className="pitch-speed predicted-highest">
        {highest.type}
      </div>
      <div className="pitch-result predicted-main">
        {(highest.value * 100).toFixed(1)}%
      </div>
      <div className="predicted-others">
        {others.map((pred, idx) => (
          <div key={idx} className="predicted-other">
            {pred.type}: {(pred.value * 100).toFixed(1)}%
          </div>
        ))}
      </div>
    </div>
  );
};

export default PredictionCard;