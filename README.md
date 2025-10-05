# pitch predictor

Pitch Predictor is a LSTM-powered MLB pitch prediction application. The application itself is a full-stack live dashboard for MLB games, and hosts a custom-built LSTM that predicts pitch types in real-time. It consists of a FastAPI backend that has endpoints for the purpose of model inference (pitch prediction) and retrieving live data, as well as a React frontend for visualization and interaction. See [demo](https://drive.google.com/file/d/1pYeBDxvNBEA97ao0HRsWrbhedOowMeVV/view?usp=sharing) in action with multiple ongoing games.

## Repository Structure

- `backend/` — FastAPI backend, serving the pitch prediction API and MLB game data endpoints.
- `data-analysis/` — Jupyter notebooks, scripts, and resources for exploratory data analysis, feature engineering, and LSTM model development/training.
- `frontend/` — React (with Node.js) frontend UI for live MLB games and pitch predictions.
- `docker-compose.yml` — Docker Compose setup for orchestrating backend and frontend.

## Getting Started

### Prerequisites

- Docker and Docker Compose

### Running with Docker Compose

#### Build and Start the Application

`docker compose --profile pitch up --build -d
`
#### Start Without Rebuilding
`docker compose --profile pitch up -d`

#### Shut Down
`docker compose --profile pitch down`

## Usage

- Access the frontend at [http://localhost:8000](http://localhost:8000) (or as configured in Docker Compose).
- The frontend displays live MLB games, scores, and real-time pitch predictions from the trained LSTM model (via backend on `localhost:8001`).
- Data analysis, and model training/testing scripts reside in `data-analysis/`.
- The actual (csv) data was not released because of Github storage constraints, but can be sent if requested.

## Data

I train my model on all pitches thrown during the 2024 MLB regular season. This data was collected on [baseballsavant](https://baseballsavant.mlb.com/). Looking at similar research and projects, as well as testing different LSTM models myself, I realized it would be more beneficial to bucket the 17 pitch data types into the four main pitch type groups defined by baseballsavant: fastball, breaking, offspeed, and other. I collect individual pitcher and batter data on baseballsavant as well. 

For the frontend, I get my data using a combination of [MLB Stats API](https://github.com/toddrob99/MLB-StatsAPI) and an [ESPN scraper](https://github.com/andr3w321/espn_scraper). I use MLB Stats API for the game card summaries and the ESPN scraper for pitch-by-pitch updates. I found MLB Stats API to be the more reliable API to request and filter data from which I use it despite the ESPN scraper being essential for live model inferencing.

## Neural Network Structure

The Long Short-Term Memory model I use takes in four primary inputs:
* At-Bat Context: Features known before the pitch has been thrown (i.e. balls, strikes, runners on bases, pitch number, score).
* Memory Context: Features of the previous pitches of the same plate appearance.
* Batter Information: Individual batter statistics from the previous (2023) MLB season.
* Pitcher Information: Individual pitcher statistics from the previous (2023) MLB season.

Individual batter and pitcher features are fed through their respective single-layered perceptron to create embeddings. The memory sequence is processed by a double-layered LSTM. The At-Bat context vector is concatenated with the batter and pitcher embeddings and passed through a neural network with a single hidden layer (activated by ReLU) resulting in an "enriched context". The LSTM's last hidden state and the "enriched context" and processed through another neural network with a single hidden layer (activated by ReLU). The output layer is applied a softmax to give a probability distribution for the four pitch types.

## Notes

* I chose an LSTM over a traditional time series statistical model because I was more intrigued by the idea of the LSTM picking up more complex, non-linear relationships between past pitches, game context, and the target pitch type. Doing some initial research on the pitch prediction models, recurrent neural networks (RNNs) were the popular choice. Additionally, the number of pitches required to be in the model's memory needed to be dynamic by the process of batter plate appearances in baseball. Finally, something that was important to me was involving individual pitcher and batter stats as embeddings into the model, something I found to be a lot easier to do with a neural network. A deep learning architecture allowed me to be more flexible with this. And given the entirety of the play-by-play data I trained this architecture on, it seemed to work well. Accuracy fell substantially when training the model on the live data that could be scraped from ESPN.
* Expanding the memory context to include the pitcher's entire game (instead of the at-bat) had very minimal effect on the model's accuracy.

## Next Steps

Further hyperparameter tuning of the model. Batch sizes, change SLPs to MLPs for player embeddings, adding LSTM layers, dropout. Adding a pitch speed regression or deep learning model would be a cool add as well.

Ideally you have models trained for individual pitchers. Given constraints on training models there might be value for training a model for each pitcher type. Classfying pitcher types given pitcher historical data could be done with a kNN algorithm.
