from fastapi import FastAPI, HTTPException
import pickle
import torch
from model import model
import logging
from typing import List, Optional
import sys

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

# Load model config/specs
with open("./model/best_pitch_predictor_lstm_meta.pkl", "rb") as f:
    meta = pickle.load(f)

# Instantiate the model using the loaded specs
loaded_mod = model.PitchPredictorLSTM(
    context_dim=meta["lstm_init_args"]['context_dim'],
    memory_dim=meta["lstm_init_args"]['memory_dim'],
    num_pitchers=len(meta["pitcher_to_id"]),
    num_batters=len(meta["batter_to_id"]),
    pitcher_embed_dim=32,
    batter_embed_dim=32,
    lstm_hidden_dim=128,
    num_pitch_types=4
)

# Load the trained weights
loaded_mod.load_state_dict(torch.load("./model/best_pitch_predictor_lstm.pth", map_location="cpu"))

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
        except (ValueError, TypeError) as e:
            error_msg = f"Error converting inputs to tensors: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Make prediction
        try:
            with torch.no_grad():
                probs = model.predict_next_pitch(loaded_mod, context_tensor, pitcher_tensor, batter_tensor, memory_tensor)
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