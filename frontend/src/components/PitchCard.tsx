import React from 'react';
import { getPitchColor } from '../utils/gameUtils';

interface PitchCardProps {
  pitch: any;
  pitchNumber: number;
}

const PitchCard: React.FC<PitchCardProps> = ({ pitch, pitchNumber }) => {
  return (
    <div className="pitch-card">
      <div className="pitch-number">#{pitchNumber}</div>
      <div className={`pitch-speed ${getPitchColor(pitch?.rslt)}`}>
        {pitch?.vlcty ? `${pitch.vlcty} mph` : 'N/A'}
      </div>
      <div className="pitch-result">
        {pitch?.dsc || 'Unknown'}
      </div>
      <div className="pitch-dsc">
        {pitch?.ptchDsc || 'Unknown'}
      </div>
    </div>
  );
};

export default PitchCard;