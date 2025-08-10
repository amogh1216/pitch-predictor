// hooks/usePrediction.ts
import { useState, useEffect } from 'react';
import { PredictedPitch, CurrentInningData, Game } from '../types/Game';

export const usePrediction = (currentInningData: CurrentInningData | null, selectedGame: Game | undefined) => {
  const [predictedPitch, setPredictedPitch] = useState<PredictedPitch | null>(null);

  const callPredictAPI = async (exportData: any) => {
    if (!exportData || !selectedGame) return;

    try {
      // Add game context to export data
      const enrichedExportData = {
        ...exportData,
        gameState: {
          ...exportData.gameState,
          inning: selectedGame.current_inning,
          inningState: selectedGame.inning_state,
          awayScore: parseInt(selectedGame.away_score as string, 10) || 0,
          homeScore: parseInt(selectedGame.home_score as string, 10) || 0,
        }
      };

      const response = await fetch("http://localhost:8001/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(enrichedExportData)
      });

      const result = await response.json();
      setPredictedPitch(result.predictions);
    } catch (error) {
      console.error('Error calling predict_pitch:', error);
      setPredictedPitch(null);
    }
  };

  useEffect(() => {
    if (currentInningData?.exportData) {
      callPredictAPI(currentInningData.exportData);
    } else {
      setPredictedPitch(null);
    }
  }, [currentInningData?.exportData, selectedGame]);

  return { predictedPitch };
};