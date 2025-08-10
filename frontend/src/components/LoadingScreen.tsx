import React from 'react';

const LoadingScreen: React.FC = () => {
  return (
    <div className="app">
      <div className="loading">
        <div className="spinner"></div>
        <p>Loading MLB games...</p>
      </div>
    </div>
  );
};

export default LoadingScreen;