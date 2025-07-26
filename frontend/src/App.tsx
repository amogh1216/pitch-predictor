import React, { useState, useEffect } from 'react';
import './App.css';

type Game = {
  game_id: number;
  espn_id: string;
  game_datetime: string;
  status: string;
  away_score: number | string;
  home_score: number | string;
  current_inning: number;
  inning_state: string;
  summary: string;
};

function App() {
  const [games, setGames] = useState<Game[]>([]);
  const [selectedGame, setSelectedGame] = useState<Game | undefined>(undefined);
  const [playByPlay, setPlayByPlay] = useState<any[]>([]);


  async function fetchPlayByPlay(espn_id: string) {
    try {
      const response = await fetch(`http://localhost:8001/api/espn_games/${espn_id}`);
      const data = await response.json();
      // Assuming the response is: { 'play-by-play': [...] }
      setPlayByPlay(data['play-by-play'] || []);
    } catch (error) {
      console.error(`Failed to fetch play-by-play for espn_id ${espn_id}:`, error);
      setPlayByPlay([]);
    }
  }

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    async function fetchGames() {

      try {
        const response = await fetch('http://localhost:8001/api/games');
        const data = await response.json();

        const espnResponse = await fetch('http://localhost:8001/api/espn_games');
        const espnGameIds: string[] = await espnResponse.json();
  
        const normalizedData = data.map((game: any, index: number) => ({
          ...game,
          espn_id: espnGameIds[index] || null,
          away_score: typeof game.away_score === 'string' ? parseInt(game.away_score) || 0 : game.away_score,
          home_score: typeof game.home_score === 'string' ? parseInt(game.home_score) || 0 : game.home_score,
          current_inning: game.current_inning === '' ? 0 : game.current_inning,
          inning_state: game.inning_state || 'N/A',
        }));
  
        console.log('Fetched and normalized games:', normalizedData);
        setGames(normalizedData);
        if (normalizedData.length > 0) {
          setSelectedGame(normalizedData[0]);
        } else {
          setSelectedGame(undefined);
        }
      } catch (error) {
        console.error('Failed to fetch games:', error);
        setGames([]);
        setSelectedGame(undefined);
      }
    }
    fetchGames();

    intervalId = setInterval(fetchGames, 30_000); // fetch every 60 seconds
    // Cleanup on unmount
    return () => clearInterval(intervalId)
    
  }, []);  

  if (!selectedGame) return <div>Loading...</div>;

  const inningIndex = selectedGame && selectedGame.current_inning ? 
    (selectedGame.current_inning - 1) * 2 + (selectedGame.inning_state.toLowerCase() === 'top' ? 0 : 1) : null;
    
  const currentInningPlayByPlay = inningIndex !== null && playByPlay.length > inningIndex ? playByPlay[inningIndex] : null;

  return (
    <div className="main-layout">
      <header className="title">Pitch Predictor</header>
      <div className="content">
        <aside className="left-panel">
          <h2>Game Details</h2>
          <div className="context-item"><strong>Summary:</strong> {selectedGame.summary}</div>
          <div className="context-item"><strong>Status:</strong> {selectedGame.status}</div>
          <div className="context-item"><strong>Date/Time:</strong> {selectedGame.game_datetime}</div>
          <div className="context-item"><strong>Inning:</strong> {selectedGame.current_inning} ({selectedGame.inning_state})</div>
          <h3>Play By Play for Inning {selectedGame?.current_inning} ({selectedGame?.inning_state})</h3>
          {currentInningPlayByPlay ? (
            <>
              <div><strong>Errors:</strong> {currentInningPlayByPlay.errors}</div>
              <div><strong>Hits:</strong> {currentInningPlayByPlay.hits}</div>
              <div><strong>Runs:</strong> {currentInningPlayByPlay.runs}</div>
              <div>
                  <strong>Plays:</strong>
                  <ul>
                    {Array.isArray(currentInningPlayByPlay.plays) ? (
                      currentInningPlayByPlay.plays.map((play: any, playIdx: number) => (
                        <li key={playIdx}>
                          {/* Print the "dsc" field of the play if it exists */}
                          <div><strong>Description:</strong> {play.dsc || "(No description)"}</div>

                          {/* Iterate over pitches if present */}
                          {Array.isArray(play.pitches) && play.pitches.length > 0 ? (
                            <ul>
                              {play.pitches.map((pitch: any, pitchIdx: number) => (
                                <li key={pitchIdx}>
                                  {/* Print all pitch info as JSON or customize */}
                                  <pre>{JSON.stringify(pitch, null, 2)}</pre>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <div>No pitches info available.</div>
                          )}
                        </li>
                      ))
                    ) : (
                      <li>No plays data available.</li>
                    )}
                  </ul>
                </div>
            </>
          ) : (
            <p>No play-by-play data available for this inning.</p>
          )}
        </aside>
        <section className="right-panel">
          <h2>Live Game Dashboard</h2>
          <div className="box-score">{selectedGame.away_score} - {selectedGame.home_score}</div>
          <div className="game-list">
            <h3>Games</h3>
            <ul>
              {games.map(game => (
                <li
                  key={game.game_id}
                  className={game.game_id === selectedGame.game_id ? 'selected' : ''}
                  onClick={() => {
                    setSelectedGame(game);
                    if (game.espn_id) {
                      fetchPlayByPlay(game.espn_id);
                    } else {
                      setPlayByPlay([]); // clear if no espn_id
                    }
                  }}
                >
                  {game.summary} ({game.away_score} - {game.home_score})
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