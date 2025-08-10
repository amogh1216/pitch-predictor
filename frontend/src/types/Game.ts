export type Game = {
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

export type PredictedPitch = {
  FAST?: number;
  OFF?: number;
  BREAK?: number;
  OTH?: number;
};

export type CurrentInningData = {
  playByPlay: any;
  bases: boolean[];
  outsCount: number;
  ballsCount: number;
  strikesCount: number;
  pitchCount: string | null;
  exportData: any;
};