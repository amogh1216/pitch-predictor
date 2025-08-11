import { useState, useEffect } from 'react';
import { CurrentInningData } from '../types/Game';

export const usePlayByPlay = (espnId: string | undefined, isActive: boolean) => {
  const [playByPlay, setPlayByPlay] = useState<any[]>([]);
  const [currentInningData, setCurrentInningData] = useState<CurrentInningData | null>(null);
  const [isPlayByPlayLoading, setIsLoading] = useState(false);

  const fetchPlayByPlay = async (espn_id: string) => {
    try {
      setIsLoading(true);
      const response = await fetch(`http://localhost:8001/api/espn_games/${espn_id}`);
      const data = await response.json();
      setPlayByPlay(data['play-by-play'] || []);
    } catch (error) {
      console.error(`Failed to fetch play-by-play for espn_id ${espn_id}:`, error);
      setPlayByPlay([]);
    } finally {
      setIsLoading(false);
    }
  };

  const processCurrentInning = (playByPlayData: any[]) => {
    if (!playByPlayData.length) {
      setCurrentInningData(null);
      return;
    }

    const currentInningPlayByPlay = playByPlayData[0];
    let bases: boolean[] = [false, false, false];
    let outsCount = 0;
    let ballsCount = 0;
    let strikesCount = 0;
    let pitchCount: string | null = null;
    let exportData: any = null;

    // Process base runners
    if (currentInningPlayByPlay?.plays?.length > 0) {
      const firstPitchWithEvents = currentInningPlayByPlay.plays[0].pitches?.find(
        (pitch: any) => pitch.evnts && pitch.evnts.onBase
      );
      if (firstPitchWithEvents) {
        bases = firstPitchWithEvents.evnts.onBase;
      }
    }

    // Count outs
    if (Array.isArray(currentInningPlayByPlay?.plays)) {
      outsCount = currentInningPlayByPlay.plays.reduce((sum: number, play: any) => {
        return sum + (typeof play.dsc === 'string' && play.dsc.toLowerCase().includes(' out ') ? 1 : 0);
      }, 0);
    }

    // Process pitch count and balls/strikes for first play
    if (Array.isArray(currentInningPlayByPlay?.plays) && currentInningPlayByPlay.plays.length > 0) {
      const play = currentInningPlayByPlay.plays[0];
      if (Array.isArray(play.pitches) && play.pitches.length > 0) {
        const mostRecentPitch = play.pitches[0];
        if (mostRecentPitch?.count) {
          pitchCount = mostRecentPitch.count.toString();
        }

        // Calculate balls and strikes
        const pitchesChronological = [...play.pitches].reverse();
        for (const pitch of pitchesChronological) {
          if (pitch.rslt === "strike" || pitch.rslt === "foul") {
            if (strikesCount < 2) strikesCount++;
          } else if (pitch.rslt === "ball") {
            ballsCount++;
          }
        }

        // Prepare export data for predictions
        exportData = {
          gameState: {
            ballsCount,
            strikesCount,
            bases,
            outsCount,
            pitchCount,
            runs: parseInt(currentInningPlayByPlay.runs as string, 10) || 0
          },
          plateAppearanceData: {
            vlcty: play.pitches.map((p: any) => p.vlcty || null),
            rslt: play.pitches.map((p: any) => p.rslt || null),
            dsc: play.pitches.map((p: any) => p.dsc || null),
            ptchDsc: play.pitches.map((p: any) => p.ptchDsc || null)
          }
        };
      }
    }

    setCurrentInningData({
      playByPlay: currentInningPlayByPlay,
      bases,
      outsCount,
      ballsCount,
      strikesCount,
      pitchCount,
      exportData
    });
  };

  // Fetch play-by-play when active and espnId is available
  useEffect(() => {
    if (!isActive || !espnId) {
      setPlayByPlay([]);
      setCurrentInningData(null);
      return;
    }

    fetchPlayByPlay(espnId);
    const intervalId = setInterval(() => fetchPlayByPlay(espnId), 10_000);
    return () => clearInterval(intervalId);
  }, [isActive, espnId]);

  // Process current inning data when playByPlay changes
  useEffect(() => {
    processCurrentInning(playByPlay);
  }, [playByPlay]);

  return {
    playByPlay,
    currentInningData,
    isPlayByPlayLoading
  };
};