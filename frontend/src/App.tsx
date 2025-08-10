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
  let [away, home] = espnName.split(' at ');
  // handle Athletics edge case
  if (away === 'Athletics Athletics') away = 'Athletics';
  if (home === 'Athletics Athletics') home = 'Athletics';
  return { away: away?.trim() || '', home: home?.trim() || '' };
}


function App() {
  const [games, setGames] = useState<Game[]>([]);
  const [selectedGame, setSelectedGame] = useState<Game | undefined>(undefined);
  const [playByPlay, setPlayByPlay] = useState<any[]>([]);
  const [isFading, setIsFading] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [bases, setBases] = useState<boolean[]>([false, false, false]);
  const [exportData, setExportData] = useState<any>(null);
  const [predictedPitch, setPredictedPitch] = useState<{
    FAST?: number;
    OFF?: number;
    BREAK?: number;
    OTH?: number;
  } | null>(null);

  const [inningIndex, setInningIndex] = useState(0);
  const [currentInningPlayByPlay, setCurrentInningPlayByPlay] = useState<any>(null);

  let pitchCount: string | null = null;
  let ballsCount = 0;
  let strikesCount = 0;
  let outsCount = 0;

  async function fetchPlayByPlay(espn_id: string) {
    try {
      setIsFading(true);
      const response = await fetch(`http://localhost:8001/api/espn_games/${espn_id}`);
      const data = await response.json();
      setPlayByPlay(data['play-by-play'] || []);

      setTimeout(() => {
        setPlayByPlay(data['play-by-play'] || []);
        setIsFading(false); // Fade in new data
      }, 200); // Duration of fade out (matches CSS, 200ms)
    } catch (error) {
      console.error(`Failed to fetch play-by-play for espn_id ${espn_id}:`, error);
      setPlayByPlay([]);
      setIsFading(false);
    }
  }

  async function fetchGames() {
    try {
      const response = await fetch('http://localhost:8001/api/games');
      const data = await response.json();

      const espnResponse = await fetch('http://localhost:8001/api/espn_games');
      //const espnGameIds: string[] = await espnResponse.json();

      const espnData = await espnResponse.json();

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

  async function callPredictAPI() {
    if (!exportData) {
      console.error('No export data available for prediction');
      return;
    }
    try {
      console.log(`exportData: ${JSON.stringify(exportData)}`);
      const response = await fetch("http://localhost:8001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(exportData)
      });
      const result = await response.json();
      console.log("Prediction result:", result);

      // Save to state
      setPredictedPitch(result.predictions);
    } catch (error) {
      console.error('Error calling predict_pitch:', error);
    }
  }


  const handleGameSelect = (game: Game) => {
    setBases([false, false, false]);
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

  const getPredictionData = () => {
    if (!predictedPitch) return null;

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

    return { highest, others };
  };

  useEffect(() => {
    let intervalId: NodeJS.Timeout;
    
    fetchGames();
    intervalId = setInterval(fetchGames, 30_000);

    if (selectedGame && selectedGame.current_inning) {
      let top = selectedGame.inning_state.toLowerCase() === 'top' ? 0 : 1;
      setInningIndex((selectedGame.current_inning - 1) * 2 + top);
      // setInningIndex(0);
    }
    else setInningIndex(0);
    console.log(`inning index: ${inningIndex} and playbyplay[0]: ${playByPlay}`);
    if (inningIndex !== null && playByPlay[0]) {
      setCurrentInningPlayByPlay(playByPlay[0]);
    }
    else setCurrentInningPlayByPlay(null);

    if (currentInningPlayByPlay?.plays && currentInningPlayByPlay.plays.length > 0) {
      // Find first pitch that has evnts with onBase
      const firstPitchWithEvents = currentInningPlayByPlay.plays[0].pitches.find(
        (pitch: any) => pitch.evnts && pitch.evnts.onBase
      );
      if (firstPitchWithEvents) {
        setBases(firstPitchWithEvents.evnts.onBase); // Should be [bool, bool, bool]
      }
    } else {
      setBases([false, false, false]);
    }

    return () => clearInterval(intervalId);
  }, [playByPlay, selectedGame]);  

  useEffect(() => {
    let intervalId: NodeJS.Timeout | undefined;

    // Only poll if details are shown and game has a valid espn_id
    if (showDetails && selectedGame?.espn_id) {
      // Fetch immediately once
      fetchPlayByPlay(selectedGame.espn_id);

      intervalId = setInterval(() => {
        fetchPlayByPlay(selectedGame.espn_id!);
        fetchGames();
      }, 10000);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [showDetails, selectedGame?.espn_id]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout | undefined;
    callPredictAPI();
  }, [exportData]);

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

  const BaseIndicator = ({ bases }: { bases: boolean[] }) => {
    // bases[0] = first base, bases[1] = second base, bases[2] = third base
    return (
      <div className="base-indicator">
        <div className={`base first-base ${bases[0] ? 'occupied' : ''}`}></div>
        <div className={`base second-base ${bases[1] ? 'occupied' : ''}`}></div>
        <div className={`base third-base ${bases[2] ? 'occupied' : ''}`}></div>
      </div>
    );
  };
    
  outsCount = Array.isArray(currentInningPlayByPlay?.plays)
    ? currentInningPlayByPlay.plays.reduce((sum: number, play: any) => {
        return sum + (typeof play.dsc === 'string' && play.dsc.toLowerCase().includes('out') ? 1 : 0);
      }, 0) : 0;

  useEffect(() => {
    if (
      Array.isArray(currentInningPlayByPlay?.plays) &&
      currentInningPlayByPlay.plays.length > 0 &&
      selectedGame
    ) {
      const play = currentInningPlayByPlay.plays[0];
      if (Array.isArray(play.pitches) && play.pitches.length > 0) {
        const gameState = {
          ballsCount,
          strikesCount,
          bases,
          outsCount,
          inning: selectedGame.current_inning,
          inningState: selectedGame.inning_state,
          pitchCount,
          awayScore: parseInt(selectedGame.away_score as string, 10) || 0,
          homeScore: parseInt(selectedGame.home_score as string, 10) || 0,
          runs: parseInt(currentInningPlayByPlay.runs as string, 10) || 0
        };
        const plateAppearanceData = {
          vlcty: play.pitches.map((p: any) => p.vlcty || null),
          rslt: play.pitches.map((p: any) => p.rslt || null),
          dsc: play.pitches.map((p: any) => p.dsc || null),
          ptchDsc: play.pitches.map((p: any) => p.ptchDsc || null)
        };
        const exportPayload = { gameState, plateAppearanceData };
        setExportData(exportPayload);
      }
    }
  }, [currentInningPlayByPlay, bases, outsCount, selectedGame]);


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
                    {selectedGame.current_inning > 0 && (
                      <span className="inning">
                        {selectedGame.inning_state} {selectedGame.current_inning}
                      </span>
                    )}
                  </div>
                </div>

                <div className={`play-by-play-section${isFading ? ' fading' : ''}`}>
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
                        <div className="stat-item">
                          <span className="stat-label">Outs</span>
                          <span className="stat-value">{outsCount}</span>
                        </div>
                      </div>

                      <div className="plays-list">
                        <h5>Recent Plays</h5>
                        <BaseIndicator bases={bases} />
                        {Array.isArray(currentInningPlayByPlay.plays) ? (
                          currentInningPlayByPlay.plays.map((play: any, playIdx: number) => {
                            //let pitchCount: string | null = null;
                            ballsCount = 0;
                            strikesCount = 0;

                            if (playIdx === 0 && Array.isArray(play.pitches) && play.pitches.length > 0) {
                              // // Get pitch count from most recent pitch (pitchIdx === 0)
                              const mostRecentPitch = play.pitches[0];
                              if (mostRecentPitch && mostRecentPitch.count) {
                                pitchCount = mostRecentPitch.count.toString();
                                // setPitchCount(mostRecentPitch.count.toString());
                              }
                              // To process pitches in chronological order
                              const pitchesChronological = [...play.pitches].reverse();

                              for (const pitch of pitchesChronological) {
                                if (pitch.rslt === "strike") {
                                  if (strikesCount < 2) strikesCount++;
                                } else if (pitch.rslt === "foul") {
                                  if (strikesCount < 2) strikesCount++;
                                } else if (pitch.rslt === "ball") {
                                  ballsCount++;
                                }
                                // Optionally, break if ballsCount === 4 or strikesCount === 3
                              }
                            }
                            return (
                              <div key={playIdx} className="play-item">
                                <div className="play-description">
                                  {play.dsc || "(No description)"}
                                  {/* Append pitch count and balls-strikes only for first play */}
                                  {playIdx === 0 && pitchCount !== null && (
                                    <> {" "} (Pitch count: {pitchCount}, Count: {ballsCount}-{strikesCount})</>
                                  )}
                                </div>

                                {Array.isArray(play.pitches) && play.pitches.length > 0 && (
                                  <div className="pitches-container">
                                    <div className="pitches-header">Pitches:</div>
                                    <div className="pitches-grid">

                                      {predictedPitch && (() => {
                                        const predictionData = getPredictionData();
                                        if (predictionData && playIdx === 0 && selectedGame.status !== "Final") {
                                          return (
                                            <div className="pitch-card predicted-pitch">
                                              <div className="pitch-number">Predicted</div>
                                              <div className="pitch-speed predicted-highest">
                                                {predictionData.highest.type}
                                              </div>
                                              <div className="pitch-result predicted-main">
                                                {(predictionData.highest.value * 100).toFixed(1)}%
                                              </div>
                                              <div className="predicted-others">
                                                {predictionData.others.map((pred, idx) => (
                                                  <div key={idx} className="predicted-other">
                                                    {pred.type}: {(pred.value * 100).toFixed(1)}%
                                                  </div>
                                                ))}
                                              </div>
                                            </div>
                                          );
                                        }
                                        else if (playIdx === 0 && selectedGame.status !== "Final") {
                                          return (
                                            <div className="pitch-card predicted-pitch">
                                                <div className="pitch-number">Predicted</div>
                                                <div className="pitch-speed predicted-highest">
                                                  Not Available
                                                </div>
                                              </div>
                                          );
                                        }
                                        return null;
                                      })()}

                                      {play.pitches.map((pitch: any, pitchIdx: number) => {
                                        const getPitchColor = (result: string) => {
                                          switch(result?.toLowerCase()) {
                                            case 'strike': return 'strike';
                                            case 'foul': return 'foul';
                                            case 'ball': return 'ball';
                                            case 'play': return 'play';
                                            default: return 'default';
                                          }
                                        };

                                        return (
                                          <div key={pitchIdx} className="pitch-card">
                                            <div className="pitch-number">#{play.pitches.length - pitchIdx}</div>
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
                                      })}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })
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