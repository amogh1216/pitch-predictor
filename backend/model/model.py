import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from tqdm import tqdm

class BaseballPitchDataset(Dataset):
    def __init__(self, df, parent_df, batter_stats_df, pitcher_stats_df, max_sequence_length=15):
        self.max_seq_len = max_sequence_length
        self.df_len = len(df)
        
         # Features known BEFORE pitch is thrown, not include 'batter', 'pitcher' ids
        self.context_features = [
            'stand_R', 'p_throws_R', 'balls', 'strikes',

            'on_3b', 'on_2b', 'on_1b', 'outs_when_up', 'inning', 'inning_topbot_Top', 'at_bat_number', 'pitch_number',
            'home_score', 'away_score', 'prev_runs_scored'
        ]

        self.parent_df = parent_df
                
        # Features from PREVIOUS pitches (memory)
        self.memory_features = [
            'release_speed', 'release_pos_y',

            'pitch_type_group_BREAK', 'pitch_type_group_FAST', 'pitch_type_group_OFF',

            'description_ball', 'description_blocked_ball', 'description_called_strike', 'description_foul', 
            'description_foul_bunt', 'description_foul_tip', 'description_hit_by_pitch', 'description_hit_into_play', 
            'description_pitchout', 'description_swinging_strike', 'description_swinging_strike_blocked', 
            'type_B', 'type_S'
        ]
        
        self.target = 'pitch_type_group'
        
        # Create player ID mappings from central df, specific to train or test df
        unique_pitchers = sorted(parent_df['pitcher'].unique())
        self.pitcher_to_id = {pid: idx for idx, pid in enumerate(unique_pitchers)}
        self.id_to_pitcher = {idx: pid for pid, idx in self.pitcher_to_id.items()}
        
        unique_batters = sorted(parent_df['batter'].unique())
        self.batter_to_id = {bid: idx for idx, bid in enumerate(unique_batters)}
        self.id_to_batter = {idx: bid for bid, idx in self.batter_to_id.items()}
        
        self.num_pitchers = len(self.pitcher_to_id)
        self.num_batters = len(self.batter_to_id)

        self.batter_stats_df = batter_stats_df #.set_index('mlbID')
        self.pitcher_stats_df = pitcher_stats_df #.set_index('mlbID')

        self.pitcher_stat_cols = ['#days', 'Age', 'G', 'GS', 'W', 'L', 'SV', 'IP', 'H', 'R', 'ER', 'BB', 
                                  'SO', 'HR', 'HBP', 'ERA', 'AB', '2B', '3B', 'IBB', 'GDP', 'SF', 'SB', 'CS', 'PO', 'BF', 'Pit', 
                                  'Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip', 'SO9', 'SO/W']
        
        self.batter_stat_cols = ['#days', 'Age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'IBB', 'SO', 'HBP', 
                                 'SH', 'SF', 'GDP', 'SB', 'CS', 'BA', 'OBP', 'SLG', 'OPS']   
        
        # Prepare encoders for other features
        self.label_encoders = {}
        self.scalers = {}

        self.pitch_type_to_idx = {}
        
        # Process data
        self.sequences = self._prepare_sequences(df)
        
    
    def __len__(self):
        return len(self.sequences)

    def _prepare_sequences(self, df):
        sequences = []

        pitch_types = ['FAST', 'OFF', 'BREAK', 'OTH']
        self.pitch_type_to_idx = {name: idx for idx, name in enumerate(pitch_types)}
        
        for plate_app_id, group in tqdm(df.groupby('plate_app_id'), desc="Processing plate appearances"):
            group = group.sort_values('pitch_number').reset_index(drop=True)
            
            if len(group) < 2:
                continue
                
            for i in range(len(group) - 1):
                current_context = group.iloc[i][self.context_features].astype(float).values

                target_pitch = self.pitch_type_to_idx[group.iloc[i][self.target]] # group.iloc[i + 1][self.target]
                
                # Get pitcher and batter IDs
                pitcher_id = self.pitcher_to_id[group.iloc[i]['pitcher']]
                batter_id = self.batter_to_id[group.iloc[i]['batter']]
                
                # Previous pitches for memory
                prev_pitches = group.iloc[:i][self.memory_features] if i > 0 else None
                
                sequences.append({
                    'context': current_context,
                    'pitcher_id': pitcher_id,
                    'batter_id': batter_id,
                    'memory_sequence': prev_pitches,
                    'target': target_pitch
                })

            # If tqdm loop is stuck at the end, exit and return sequences
            if not hasattr(self, '_last_plate_app_id'):
                self._last_plate_app_id = None
            if self._last_plate_app_id == plate_app_id:
                print("Detected stuck tqdm loop. Exiting and returning sequences.")
                return sequences
            self._last_plate_app_id = plate_app_id
        
        return sequences
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Context features
        context = torch.FloatTensor(sequence['context'])
        
        # Player IDs as integers (not tensors yet)
        pitcher_id_scalar = sequence['pitcher_id']
        batter_id_scalar = sequence['batter_id']

        # Player IDs
        pitcher_id = torch.LongTensor([pitcher_id_scalar])
        batter_id = torch.LongTensor([batter_id_scalar])
        
        # Memory sequence (same as before)
        if sequence['memory_sequence'] is not None:
            memory = sequence['memory_sequence'].values
            if len(memory) > self.max_seq_len:
                memory = memory[-self.max_seq_len:]
            else:
                padding = np.zeros((self.max_seq_len - len(memory), len(self.memory_features)))
                memory = np.vstack([padding, memory])
        else:
            memory = np.zeros((self.max_seq_len, len(self.memory_features)))
        
        memory_tensor = torch.FloatTensor(memory)
        target = torch.LongTensor([sequence['target']])

        # added
        pitcher_row = self.pitcher_stats_df[self.pitcher_stats_df['mlbID'] == pitcher_id_scalar]
        if not pitcher_row.empty:
            pitcher_stats_vec = torch.FloatTensor(pitcher_row[self.pitcher_stat_cols].values[0])
        else:
            pitcher_stats_vec = torch.zeros(len(self.pitcher_stat_cols))

        batter_row = self.batter_stats_df[self.batter_stats_df['mlbID'] == batter_id_scalar]
        if not batter_row.empty:
            batter_stats_vec = torch.FloatTensor(batter_row[self.batter_stat_cols].values[0])
        else:
            batter_stats_vec = torch.zeros(len(self.batter_stat_cols))
        
        return context, pitcher_id, batter_id, memory_tensor, pitcher_stats_vec, batter_stats_vec, target

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class PitchPredictorLSTM(nn.Module):
    def __init__(self, context_dim, memory_dim, num_pitchers, num_batters, pitcher_stats_dim=37, batter_stats_dim=24,
                 pitcher_embed_dim=16, batter_embed_dim=16, lstm_hidden_dim=128, 
                 num_pitch_types=10, lstm_layers=2):
        super(PitchPredictorLSTM, self).__init__()
        
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers

        # Pass player features through an MLP to create their embedding:
        self.pitcher_feat_embed = nn.Sequential(
            nn.Linear(pitcher_stats_dim, 32),
            nn.ReLU(),
            nn.Linear(32, pitcher_embed_dim)
        )
        self.batter_feat_embed = nn.Sequential(
            nn.Linear(batter_stats_dim, 32),
            nn.ReLU(),
            nn.Linear(32, batter_embed_dim)
        )
        
        # LSTM for processing previous pitches
        self.lstm = nn.LSTM(
            input_size=memory_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # Context processing (now includes embeddings)
        context_input_dim = context_dim + pitcher_embed_dim + batter_embed_dim
        self.context_fc = nn.Sequential(
            nn.Linear(context_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Combine LSTM output and enriched context
        self.combined_fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_pitch_types)
        )
                
    def forward(self, context, pitcher_ids, batter_ids, memory_sequence, pitcher_stats, batter_stats):
        batch_size = memory_sequence.size(0)

        # Instead of nn.Embedding lookup, embed the stats vector
        pitcher_embeds = self.pitcher_feat_embed(pitcher_stats)
        batter_embeds = self.batter_feat_embed(batter_stats)
        
        # Process memory sequence through LSTM
        lstm_out, (hidden, cell) = self.lstm(memory_sequence)
        lstm_final = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)
        
        # Combine context with player embeddings
        enriched_context = torch.cat([context, pitcher_embeds, batter_embeds], dim=1)
        context_out = self.context_fc(enriched_context)  # (batch_size, 64)
        
        # Combine LSTM memory and enriched context
        combined = torch.cat([lstm_final, context_out], dim=1)
        
        # Final prediction
        output = self.combined_fc(combined)
        
        return output


def load_training_data(csv, hasDataSet, batter_stats_csv, pitcher_stats_csv):
    print('reading batter csv')
    batter_stats_df = pd.read_csv(batter_stats_csv)
    print('read batter csv')

    print('reading pitcher csv')
    pitcher_stats_df = pd.read_csv(pitcher_stats_csv)
    print('read pitcher csv')
    
    if not hasDataSet:
        # Load your data
        print('reading dataset csv')
        df = pd.read_csv(csv)
        print('read dataset csv')

        # Shuffle and split the DataFrame into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

        print('processing train dataset')
        train_dataset = BaseballPitchDataset(train_df, df, batter_stats_df, pitcher_stats_df)
        print('processing val dataset')
        val_dataset = BaseballPitchDataset(val_df, df, batter_stats_df, pitcher_stats_df)
        print('finished processing datasets')

        # Save the train and val datasets using pickle
        with open('./data/train_dataset_emb.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open('./data/val_dataset_emb.pkl', 'wb') as f:
            pickle.dump(val_dataset, f)
        
        return train_dataset, val_dataset
    else:
        # Load the train and val datasets from pickle
        print('loading train_dataset from pickle')
        with open('./data/train_dataset_emb.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        print('loading val_dataset from pickle')
        with open('./data/val_dataset_emb.pkl', 'rb') as f:
            val_dataset = pickle.load(f)
        
        return train_dataset, val_dataset

# Updated training setup
def train_model(train_dataset, val_dataset):

    
    # Prepare relevant data to save
    save_data = {
        "pitcher_to_id": train_dataset.pitcher_to_id,
        "batter_to_id": train_dataset.batter_to_id,
        "pitch_type_to_idx": train_dataset.pitch_type_to_idx,
        "lstm_init_args": {
            "context_dim": len(train_dataset.context_features),
            "memory_dim": len(train_dataset.memory_features),
            "num_pitchers": train_dataset.num_pitchers,
            "num_batters": train_dataset.num_batters,
            "pitcher_embed_dim": 16,
            "batter_embed_dim": 16,
            "lstm_hidden_dim": 128,
            "num_pitch_types": 4
        }
    }

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print('initting model architecture')
    # Initialize model with embedding dimensions
    model = PitchPredictorLSTM(
        context_dim=len(train_dataset.context_features),
        memory_dim=len(train_dataset.memory_features),
        num_pitchers=train_dataset.num_pitchers,
        num_batters=train_dataset.num_batters,
        pitcher_stats_dim=37, 
        batter_stats_dim=24,
        pitcher_embed_dim=16,  # Reasonable size
        batter_embed_dim=16,   # Reasonable size
        lstm_hidden_dim=128,
        num_pitch_types=4 # ['FAST', 'OFF', 'BREAK', 'OTH']
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print('starting training loop')

    # Custom EarlyStopping implementation with patience 10
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    n_epochs = 100

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for context, pitcher_ids, batter_ids, memory_seq, pitcher_stats, batter_stats, targets in train_dataloader:
            #print('lalal')
            optimizer.zero_grad()
            
            outputs = model(context, pitcher_ids, batter_ids, memory_seq, pitcher_stats, batter_stats)
            loss = criterion(outputs, targets.squeeze())
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for context, pitcher_ids, batter_ids, memory_seq, pitcher_stats, batter_stats, targets in val_dataloader:
                outputs = model(context, pitcher_ids, batter_ids, memory_seq, pitcher_stats, batter_stats)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_dataloader)

        print(f'Epoch {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Custom early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "pitch_predictor_lstm1.pth")
            with open("pitch_predictor_lstm_meta1.pkl", "wb") as f:
                pickle.dump(save_data, f)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs. Best Val Loss: {best_val_loss:.4f}")
            model.load_state_dict(torch.load("pitch_predictor_lstm1.pth"))
            return model

    # Save the trained model and metadata at the end as well
    torch.save(model.state_dict(), "pitch_predictor_lstm1.pth")
    with open("pitch_predictor_lstm_meta1.pkl", "wb") as f:
        pickle.dump(save_data, f)
    return model

# For inference
def predict_next_pitch(model, current_context, pitcher_id, batter_id, previous_pitches_in_pa, pitcher_stats, batter_stats):
    model.eval()
    with torch.no_grad():
        
        context_tensor = torch.FloatTensor(current_context).unsqueeze(0)
        pitcher_tensor = torch.LongTensor([pitcher_id])
        batter_tensor = torch.LongTensor([batter_id])
        memory_tensor = torch.FloatTensor(previous_pitches_in_pa).unsqueeze(0)
        
        prediction = model(context_tensor, pitcher_tensor, batter_tensor, memory_tensor, pitcher_stats, batter_stats)
        probabilities = torch.softmax(prediction, dim=1)
        
        return probabilities.numpy()
    