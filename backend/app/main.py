from fastapi import FastAPI, HTTPException
import pickle
import torch
from model import model
import logging
from typing import List, Optional
import sys

import statsapi
import espn_scraper as espn
from datetime import date
from datetime import datetime
import pytz

from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],  # Or ["*"] for all origins (dev only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model config/specs
with open("./model/pitch_predictor_lstm_meta1.pkl", "rb") as f:
    meta = pickle.load(f)

# Instantiate the model using the loaded specs
loaded_mod = model.PitchPredictorLSTM(
    context_dim=meta["lstm_init_args"]['context_dim'],
    memory_dim=meta["lstm_init_args"]['memory_dim'],
    num_pitchers=len(meta["pitcher_to_id"]),
    num_batters=len(meta["batter_to_id"]),
    pitcher_embed_dim=16,
    batter_embed_dim=16,
    lstm_hidden_dim=128,
    num_pitch_types=4
)

# Load the trained weights
loaded_mod.load_state_dict(torch.load("./model/pitch_predictor_lstm1.pth", map_location="cpu"))

@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Pitch Predictor API is running!"}


@app.post("/predict")
def predict_pitch(current_context: List[float], pitcher_id: int, batter_id: int, previous_pitches_in_pa: Optional[List[List[float]]] = None) -> dict:
    """
    Predict the next pitch type probabilities based on the given context.
    
    Args:
        current_context: List of context features in the order expected by the model
        pitcher_id: MLB ID of the pitcher
        batter_id: MLB ID of the batter
        previous_pitches_in_pa: Optional list of previous pitches features in the plate appearance
    
    Returns:
        dict: Mapping of pitch types to their predicted probabilities
    
    Raises:
        HTTPException: If there are any errors during prediction
    """
    try:
        # Validate input dimensions
        expected_context_features = meta["lstm_init_args"]['context_dim']
        if len(current_context) != expected_context_features:
            error_msg = f"Expected {expected_context_features} context features, got {len(current_context)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Check if pitcher exists in our model's known pitchers
        if pitcher_id not in meta["pitcher_to_id"]:
            logger.warning(f"Unknown pitcher_id: {pitcher_id}, defaulting to index 0")

        # Check if batter exists in our model's known batters
        if batter_id not in meta["batter_to_id"]:
            logger.warning(f"Unknown batter_id: {batter_id}, defaulting to index 0")

        # Validate previous pitches if provided
        if previous_pitches_in_pa:
            expected_memory_features = meta["lstm_init_args"]['memory_dim']
            for i, pitch in enumerate(previous_pitches_in_pa):
                if len(pitch) != expected_memory_features:
                    error_msg = f"Previous pitch at index {i} has {len(pitch)} features, expected {expected_memory_features}"
                    logger.error(error_msg)
                    raise HTTPException(status_code=400, detail=error_msg)

        # Convert inputs to tensors
        try:
            context_tensor = torch.tensor(current_context, dtype=torch.float32)
            pitcher_tensor = [meta["pitcher_to_id"][pitcher_id]]
            batter_tensor = [meta["batter_to_id"][batter_id]]

            if previous_pitches_in_pa:
                memory_tensor = torch.tensor(previous_pitches_in_pa, dtype=torch.float32)
            else:
                memory_tensor = None
            
            # load pitcher, batter data
            batter_df = pd.read_csv('./data-analysis/data/batting_stats_23.csv')
            pitcher_df = pd.read_csv('./data-analysis/data/pitching_stats_23.csv')

            pitcher_stat_cols = ['#days', 'Age', 'G', 'GS', 'W', 'L', 'SV', 'IP', 'H', 'R', 'ER', 'BB', 
                                            'SO', 'HR', 'HBP', 'ERA', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS', 'PO', 'BF', 'Pit', 
                                            'Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip', 'SO9', 'SO/W']
                    
            batter_stat_cols = ['#days', 'Age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 
                                        'SH', 'SF', 'GDP', 'SB', 'CS', 'BA', 'OBP', 'SLG', 'OPS']

            # added
            #print(self.pitcher_stats_df.columns)
            pitcher_row = pitcher_df[pitcher_df['mlbID'] == 506433]
            if not pitcher_row.empty:
                pitcher_stats_vec = torch.FloatTensor(pitcher_row[pitcher_stat_cols].values[0])
            else:
                pitcher_stats_vec = torch.zeros(len(pitcher_stat_cols))

            batter_row = batter_df[batter_df['mlbID'] == 518595]
            if not batter_row.empty:
                batter_stats_vec = torch.FloatTensor(batter_row[batter_stat_cols].values[0])
            else:
                batter_stats_vec = torch.zeros(len(batter_stat_cols))

        except (ValueError, TypeError) as e:
            error_msg = f"Error converting inputs to tensors: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Make prediction
        try:
            with torch.no_grad():
                probs = model.predict_next_pitch(loaded_mod, context_tensor, pitcher_tensor, batter_tensor, memory_tensor, 
                                                 pitcher_stats=pitcher_stats_vec.unsqueeze(0), batter_stats=batter_stats_vec.unsqueeze(0))
        except Exception as e:
            error_msg = f"Model prediction failed: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

        # Map the predicted probabilities to pitch type names
        idx_to_pitch_type = {v: k for k, v in meta['pitch_type_to_idx'].items()}
        pitch_probs = {idx_to_pitch_type[i]: float(probs[0, i]) for i in range(probs.shape[1])}
        
        # Log successful prediction
        logger.info(f"Successfully predicted pitch probabilities for pitcher {pitcher_id} and batter {batter_id}")
        
        return {"predictions": pitch_probs}
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error during prediction: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)
    

"""
FASTAPI endpoints for LIVE GAME DATA
"""
todays_statsapi_ids = []
todays_espn_ids = []

tz = pytz.timezone('America/Los_Angeles')
today = datetime.now(tz)

'''
date format: YYYY-MM-DD
gets todays games
'''
@app.get("/api/games")
def get_scores_statsapi():
    sched = statsapi.schedule(date=str(today)[:10])
    sched_filter = []
    todays_statsapi_ids = []
    try:
        for game in sched:
            sched_filter.append({
                "game_id": game.get('game_id'),
                "game_datetime": game.get('game_datetime'),
                "status": game.get('status'),
                "away_score": game.get('away_score'),
                "home_score": game.get('home_score'),
                "current_inning": game.get('current_inning'),
                "inning_state": game.get('inning_state'),
                "summary": game.get('summary')
            })
            todays_statsapi_ids.append(game.get('game_id'))
    except Exception as e:
        return {"error": e}
    return sched_filter

'''
has no use as of yet
'''
@app.get("/api/game/{id}")
def get_game(id: int):
    return {"game_info": id}

'''
Returns today's game ids from ESPN website.
Format date in YYYYMMDD
'''
@app.get("/api/espn_games")
def get_espn_game_ids():
    ids = []
    names = []
    # format date correctly
    date = str(today).replace('-', '')[:8]
    print(date)
    try:
        
        games = espn.get_url(f'https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={date}')['events']
        for game in games:
            if (today.day == datetime.fromisoformat(game['date']).day):
                print(game['date'])
            ids.append(game['id'])
            names.append(game['name'])
    except Exception as e:
        print(f"An exception as occured: {e}")
        return {"error": "{e}"}
    
    todays_espn_ids = ids
    todays_espn_names = names
    return {"ids": todays_espn_ids, "names": todays_espn_names}

'''
returns play by play data for a given game id
'''
@app.get("/api/espn_games/{id}")
def get_espn_game_pbp(id: str):
    try:
         game = espn.get_url(f"https://www.espn.com/mlb/playbyplay/_/gameId/{id}&_xhr=1")
         return {'play-by-play': game['page']['content']['gamepackage']['pbp']}
    except Exception as e:
        return {"error": f"{e}"}
