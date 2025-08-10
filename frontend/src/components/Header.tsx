import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="header">
      <h1 className="title">MLB Live Dashboard</h1>
      <div className="live-indicator">
        <span className="pulse-dot"></span>
        LIVE
      </div>
    </header>
  );
};

export default Header;