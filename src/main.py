from fastapi import FastAPI, Query
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd

from preprocessing_utils import clean_categories

app = FastAPI()

# Load model for EPL goals
model_epl = joblib.load("eplgoalsmodel_rf.pkl")

# Load model for Messi goals
model_messi = joblib.load("messigoalsmodel_rf.pkl")

# Loead model for EPL matches
model_eplmatches = joblib.load("eplmatches5ymodel_rf.pkl")


# Pydantic model
class eplgoaldata(BaseModel):
    match_period: int
    minute_in_half: int	
    possession_team: str
    play_pattern: str
    position: str
    x: float	
    y: float

# Model wrapper
def epl_goalsmodel(match_period, minute_in_half, possession_team, play_pattern, position, x,y):

    columns = ['match_period', 'minute_in_half', 'possession_team', 'play_pattern', 'position','x','y']
    features = [match_period, minute_in_half, possession_team, play_pattern, position, x, y]

    df = pd.DataFrame([features], columns=columns)
    df=clean_categories(df)
    
    prediction = model_epl.predict_proba(df)[0][1]
    return prediction

# Pydantic model
class messigoaldata(BaseModel):
    match_period: int
    minute_in_half: int		
    play_pattern: str
    under_pressure: bool
    x: float	
    y: float


# Model wrapper
def messi_goalsmodel(match_period, minute_in_half, play_pattern, under_pressure, x,y):

    columns = ['match_period', 'minute_in_half', 'play_pattern', 'under_pressure','x','y']
    features = [match_period, minute_in_half, play_pattern, under_pressure, x, y]

    df = pd.DataFrame([features], columns=columns)
    df=clean_categories(df)
    
    prediction = model_messi.predict_proba(df)[0][1]
    return prediction

# Pydantic model
class eploutcomedata(BaseModel):
    position_away: float
    position_home: float
    temperature_day: float
    wind_speed: float	
    humidity: float
    pressure: float
    clouds: float
    name: str
    time_of_day: str

# Model wrapper
def epl_outcomemodel(position_away, position_home, temperature_day, wind_speed, humidity, pressure, clouds, name, time_of_day):

    columns = ['position_away', 'position_home', 'temperature_day', 'wind_speed', 'humidity', 'pressure', 'clouds', 'name', 'time_of_day']
    features = [position_away, position_home, temperature_day, wind_speed, humidity, pressure, clouds, name, time_of_day]

    df = pd.DataFrame([features], columns=columns)
    df=clean_categories(df)
    
    prediction = model_eplmatches.predict_proba(df)[0][1]
    return prediction

# Prediction route for EPL goals
@app.post("/predict/goals/epl")
def predict_eplgoals(data: eplgoaldata):
    prediction = epl_goalsmodel(data.match_period, data.minute_in_half, data.possession_team, data.play_pattern, data.position, data.x, data.y)
    
    return {"prediction": prediction}

# Prediction route for Messi goals
@app.post("/predict/goals/messi")
def predict_messigoals(data: messigoaldata):
    prediction = messi_goalsmodel(data.match_period, data.minute_in_half, data.play_pattern, data.under_pressure, data.x, data.y)
    
    return {"prediction": prediction}

# Prediction route for EPL match outcomes
@app.post("/predict/matchoutcome/epl")
def predict_eplmatchoutcome(data: eploutcomedata):
    prediction = epl_outcomemodel(data.position_away, data.position_home, data.temperature_day, data.wind_speed, data.humidity, data.pressure, data.clouds, data.name, data.time_of_day)
    
    return {"prediction": prediction}
