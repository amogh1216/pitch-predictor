import React from 'react';

interface BaseIndicatorProps {
  bases: boolean[];
}

const BaseIndicator: React.FC<BaseIndicatorProps> = ({ bases }) => {
  return (
    <div className="base-indicator">
      <div className={`base first-base ${bases[0] ? 'occupied' : ''}`}></div>
      <div className={`base second-base ${bases[1] ? 'occupied' : ''}`}></div>
      <div className={`base third-base ${bases[2] ? 'occupied' : ''}`}></div>
    </div>
  );
};

export default BaseIndicator;