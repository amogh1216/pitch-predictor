import React, { useState } from 'react';
import './App.css';

const games = [
  { id: 1, name: 'Yankees vs Red Sox', boxScore: 'Box score for Yankees vs Red Sox', context: { pitcher: 'Cole', batter: 'Devers', pitchCount: '2-1', pitchHistory: ['Fastball', 'Slider', 'Curve'], prediction: { Fastball: 0.5, Slider: 0.3, Curve: 0.2 } } },
  { id: 2, name: 'Dodgers vs Giants', boxScore: 'Box score for Dodgers vs Giants', context: { pitcher: 'Kershaw', batter: 'Crawford', pitchCount: '1-2', pitchHistory: ['Slider', 'Slider', 'Fastball'], prediction: { Fastball: 0.4, Slider: 0.4, Curve: 0.2 } } },
];

function App() {
  const [selectedGame, setSelectedGame] = useState(games[0]);

  return (
    <div className="main-layout">
      <header className="title">Pitch Predictor</header>
      <div className="content">
        <aside className="left-panel">
          <h2>Pitch Predictor</h2>
          <div className="context-item"><strong>Pitcher:</strong> {selectedGame.context.pitcher}</div>
          <div className="context-item"><strong>Batter:</strong> {selectedGame.context.batter}</div>
          <div className="context-item"><strong>Pitch Count:</strong> {selectedGame.context.pitchCount}</div>
          <div className="context-item"><strong>Pitch History:</strong>
            <ul>
              {selectedGame.context.pitchHistory.map((pitch, idx) => (
                <li key={idx}>{pitch}</li>
              ))}
            </ul>
          </div>
          <div className="context-item"><strong>Pitch Prediction:</strong>
            <ul>
              {Object.entries(selectedGame.context.prediction).map(([pitch, prob]) => (
                <li key={pitch}>{pitch}: {((prob as number) * 100).toFixed(0)}%</li>
              ))}
            </ul>
          </div>
        </aside>
        <section className="right-panel">
          <h2>Live Game Dashboard</h2>
          <div className="box-score">{selectedGame.boxScore}</div>
          <div className="game-list">
            <h3>Games</h3>
            <ul>
              {games.map(game => (
                <li key={game.id} className={game.id === selectedGame.id ? 'selected' : ''} onClick={() => setSelectedGame(game)}>
                  {game.name}
                </li>
              ))}
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}

export default App;
