import { useState, useEffect } from 'react';
import { Game } from '../types/Game';
import { parseGameSummary, parseEspnName } from '../utils/gameUtils';

export const useGameData = () => {
  const [games, setGames] = useState<Game[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchGames = async () => {
    try {
      const [response, espnResponse] = await Promise.all([
        fetch('http://localhost:8001/api/games'),
        fetch('http://localhost:8001/api/espn_games')
      ]);

      const data = await response.json();
      const espnData = await espnResponse.json();

      const normalizedData = data.map((game: any) => {
        const { away: gameAway, home: gameHome } = parseGameSummary(game.summary);
        
        let matchedEspnId: string | null = null;
        let matchedEspnName: string | null = null;

        // Find matching ESPN game
        for (let i = 0; i < espnData.names.length; i++) {
          const { away: espnAway, home: espnHome } = parseEspnName(espnData.names[i]);
          if (
            espnAway && espnHome &&
            gameAway.toLowerCase() === espnAway.toLowerCase() &&
            gameHome.toLowerCase() === espnHome.toLowerCase()
          ) {
            matchedEspnId = espnData.ids[i];
            matchedEspnName = espnData.names[i];
            break;
          }
        }

        return {
          ...game,
          summary: `${gameAway} @ ${gameHome}`,
          espn_id: matchedEspnId,
          espn_name: matchedEspnName,
          away_score: typeof game.away_score === 'string' ? parseInt(game.away_score) || 0 : game.away_score,
          home_score: typeof game.home_score === 'string' ? parseInt(game.home_score) || 0 : game.home_score,
          current_inning: game.current_inning === '' ? 0 : game.current_inning,
          inning_state: game.inning_state || 'N/A',
        };
      });

      setGames(normalizedData);
    } catch (error) {
      console.error('Failed to fetch games:', error);
      setGames([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchGames();
    const intervalId = setInterval(fetchGames, 30_000);
    return () => clearInterval(intervalId);
  }, []);

  return { games, isLoading, refetchGames: fetchGames };
};