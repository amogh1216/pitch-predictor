import React, { useState, useEffect } from 'react';
import './App.css';

type Game = {
  game_id: number;
  espn_id: string;
  espn_name: string;
  game_datetime: string;
  status: string;
  away_score: number | string;
  home_score: number | string;
  current_inning: number;
  inning_state: string;
  summary: string;
};

function parseGameSummary(summary: string) {
  // Remove prefix date and ' - '
  // Example: "2025-08-05 - San Francisco Giants @ Pittsburgh Pirates (Pre-Game)"
  const teamsPart = summary.split(' - ')[1];  // "San Francisco Giants @ Pittsburgh Pirates (Pre-Game)"
  if (!teamsPart) return { away: '', home: '' };
  
  // Remove anything after '('
  //const cleanTeams = teamsPart.split(' (')[0];  // "San Francisco Giants @ Pittsburgh Pirates"
  const cleanTeams = teamsPart.replace(/\s*\([^)]+\)\s*$/, '').trim();

  // Split around ' @ '
  const [away, home] = cleanTeams.split(' @ ');
  return { away: away?.replace(/\s*\(\d+\)/g, '').trim() || '', home: home?.replace(/\s*\(\d+\)/g, '').trim() || '' };
}

function parseEspnName(espnName: string) {
  // Example: "Houston Astros at Miami Marlins"
  const [away, home] = espnName.split(' at ');
  return { away: away?.trim() || '', home: home?.trim() || '' };
}


function App() {
  const [games, setGames] = useState<Game[]>([]);
  const [selectedGame, setSelectedGame] = useState<Game | undefined>(undefined);
  const [playByPlay, setPlayByPlay] = useState<any[]>([]);
  const [showDetails, setShowDetails] = useState(false);

  async function fetchPlayByPlay(espn_id: string) {
    try {
      const response = await fetch(`http://localhost:8001/api/espn_games/${espn_id}`);
      const data = await response.json();
      setPlayByPlay(data['play-by-play'] || []);
    } catch (error) {
      console.error(`Failed to fetch play-by-play for espn_id ${espn_id}:`, error);
      setPlayByPlay([]);
    }
  }

  const handleGameSelect = (game: Game) => {
    if (selectedGame?.game_id === game.game_id && showDetails) {
      // If clicking the same game that's already selected, toggle details
      setShowDetails(false);
    } else {
      setSelectedGame(game);
      setShowDetails(true);
      if (game.espn_id) {
        fetchPlayByPlay(game.espn_id);
      } else {
        setPlayByPlay([]);
      }
    }
  };

  const closeDetails = () => {
    setShowDetails(false);
  };

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    async function fetchGames() {
      try {
        const response = await fetch('http://localhost:8001/api/games');
        const data = await response.json();

        const espnResponse = await fetch('http://localhost:8001/api/espn_games');
        //const espnGameIds: string[] = await espnResponse.json();

        const espnData = await espnResponse.json();
  
        // const normalizedData = data.map((game: any, index: number) => ({
        //   ...game,
        //   espn_id: espnGameIds[index] || null,
        //   away_score: typeof game.away_score === 'string' ? parseInt(game.away_score) || 0 : game.away_score,
        //   home_score: typeof game.home_score === 'string' ? parseInt(game.home_score) || 0 : game.home_score,
        //   current_inning: game.current_inning === '' ? 0 : game.current_inning,
        //   inning_state: game.inning_state || 'N/A',
        // }));

        const normalizedData = data.map((game: any, index: number) => {
          const espnName = espnData.names[index];
          const espnId = espnData.ids[index];
          
          const { away: gameAway, home: gameHome } = parseGameSummary(game.summary);
          const { away: espnAway, home: espnHome } = parseEspnName(espnName);
          const teamsPart = `${gameAway} @ ${gameHome}`;

          // Compare team names (simple case-sensitive exact match; can be improved with normalization)
          // const isMatch = (gameAway.toLowerCase() === espnAway.toLowerCase()) && (gameHome.toLowerCase() === espnHome.toLowerCase());

          let matchedEspnId: string | null = null;
          let matchedEspnName: string | null = null;

          // Search through ALL espnData names/ids
          for (let i = 0; i < espnData.names.length; i++) {
            const { away: espnAway, home: espnHome } = parseEspnName(espnData.names[i]);
            if (
              espnAway && espnHome &&
              gameAway.toLowerCase() === espnAway.toLowerCase() &&
              gameHome.toLowerCase() === espnHome.toLowerCase()
            ) {
              matchedEspnId = espnData.ids[i];
              matchedEspnName = espnData.names[i];
              break; // Stop at the first match
            }
          }

          return {
            ...game,
            summary: teamsPart,
            espn_id: matchedEspnId,
            espn_name: matchedEspnName,
            away_score: typeof game.away_score === 'string' ? parseInt(game.away_score) || 0 : game.away_score,
            home_score: typeof game.home_score === 'string' ? parseInt(game.home_score) || 0 : game.home_score,
            current_inning: game.current_inning === '' ? 0 : game.current_inning,
            inning_state: game.inning_state || 'N/A',
          };
        });
  
        console.log('Fetched and normalized games:', normalizedData);
        setGames(normalizedData);
      } catch (error) {
        console.error('Failed to fetch games:', error);
        setGames([]);
      }
    }
    
    fetchGames();
    intervalId = setInterval(fetchGames, 30_000);
    
    return () => clearInterval(intervalId);
  }, []);  

  const formatDateTime = (dateTime: string) => {
    try {
      return new Date(dateTime).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      });
    } catch {
      return dateTime;
    }
  };

  const getStatusColor = (status: string) => {
    const lowerStatus = status.toLowerCase();
    if (lowerStatus.includes('live') || lowerStatus.includes('in progress')) return 'live';
    if (lowerStatus.includes('final')) return 'final';
    return 'scheduled';
  };

  const inningIndex = selectedGame && selectedGame.current_inning ? 
    (selectedGame.current_inning - 1) * 2 + (selectedGame.inning_state.toLowerCase() === 'top' ? 0 : 1) : null;
    
  const currentInningPlayByPlay = inningIndex !== null && playByPlay[0] ? playByPlay[0] : null;

  if (games.length === 0) {
    return (
      <div className="app">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading MLB games...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <h1 className="title">MLB Live Dashboard</h1>
        <div className="live-indicator">
          <span className="pulse-dot"></span>
          LIVE
        </div>
      </header>

      <main className="main">
        <div className="games-grid">
          {games.map((game) => (
            <div
              key={game.game_id}
              className={`game-card ${selectedGame?.game_id === game.game_id ? 'selected' : ''}`}
              onClick={() => handleGameSelect(game)}
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
          ))}
        </div>

        {/* Details Panel Overlay */}
        <div className={`details-overlay ${showDetails ? 'active' : ''}`}>
          <div className="details-panel">
            <div className="details-header">
              <h2>Game Details</h2>
              <button className="close-btn" onClick={closeDetails}>×</button>
            </div>
            
            {selectedGame && (
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
                    <span className="espn_name">
                      {formatDateTime(selectedGame.espn_name)}
                    </span>
                    {selectedGame.current_inning > 0 && (
                      <span className="inning">
                        {selectedGame.inning_state} {selectedGame.current_inning}
                      </span>
                    )}
                  </div>
                </div>

                <div className="play-by-play-section">
                  <h4>Play-by-Play: Inning {selectedGame.current_inning} ({selectedGame.inning_state})</h4>
                  {currentInningPlayByPlay ? (
                    <div className="inning-stats">
                      <div className="stat-grid">
                        <div className="stat-item">
                          <span className="stat-label">Runs</span>
                          <span className="stat-value">{currentInningPlayByPlay.runs}</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-label">Hits</span>
                          <span className="stat-value">{currentInningPlayByPlay.hits}</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-label">Errors</span>
                          <span className="stat-value">{currentInningPlayByPlay.errors}</span>
                        </div>
                      </div>

                      <div className="plays-list">
                        <h5>Recent Plays</h5>
                        {Array.isArray(currentInningPlayByPlay.plays) ? (
                          currentInningPlayByPlay.plays.map((play: any, playIdx: number) => (
                            <div key={playIdx} className="play-item">
                              <div className="play-description">
                                {play.dsc || "(No description)"}
                              </div>
                              {Array.isArray(play.pitches) && play.pitches.length > 0 && (
                                <div className="pitches-container">
                                  <div className="pitches-header">Pitches:</div>
                                  <div className="pitches-list">
                                    {play.pitches.map((pitch: any, pitchIdx: number) => (
                                      <div key={pitchIdx} className="pitch-item">
                                        <pre className="pitch-data">
                                          {JSON.stringify(pitch, null, 2)}
                                        </pre>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          ))
                        ) : (
                          <p className="no-data">No plays data available.</p>
                        )}
                      </div>
                    </div>
                  ) : (
                    <p className="no-data">No play-by-play data available for this inning.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;