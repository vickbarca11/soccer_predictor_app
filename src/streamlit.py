import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import numpy as np
import requests


def clean_categories(X):
    return X.applymap(lambda x: x.lower().replace(" ", "_") if isinstance(x, str) else x)

# Set up UI
st.set_page_config(page_title="Shot Outcome Predictor", layout="centered")

st.sidebar.markdown("### 📕  Menu")
option = st.sidebar.selectbox("", ['Introduction', 'Goal Probability', 'Win Probability', 'Analysis Key Points'])

if option == "Introduction":
    st.title("⚽ Soccer Prediction Tool")

    st.write("**Working on it...Coming Soon**")

if option == 'Goal Probability':
    image_display = st.empty()
    league_selection = st.selectbox(" _**Choose which league will be analyzed:**_", ['English Premier League', 'Bonus: Messi'])
    if league_selection == 'English Premier League':
        image = Image.open('../img/epl.jpg')
        image_display.image(image)

        # Load trained model pipeline
        model = joblib.load("eplgoalsmodel_rf.pkl")
        
        st.markdown("###### The parameters below change the probability of scoring a goal. Adjust different parameters to see how the probability of scoring changes. Scoring probability is shown at the bottom of the page.")
        
        ## Checking this box allows the user to see the feature importances
        important_features = st.checkbox("_Check this box if you would like to see how the parameters below influence the changes in the model (it will not affect the model's prediction if you click this):_", value=False)
        if important_features == True:
            ## Feature Importances
            numeric_features = ['match_period', 'minute_in_half', 'x', 'y']
            categorical_features = ['position', 'possession_team', 'play_pattern']

            importances = model.named_steps['classifier'].feature_importances_
            feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features))
            
            # Clean feature names
            def clean_name(name):
                name = name.lower()
                if 'play_pattern_' in name:
                    name = name.replace('play_pattern_', '')
                elif 'possession_team_' in name:
                    name = name.replace('possession_team_', '')
                elif 'position_' in name:
                    name = name.replace('position_', '')
                return name.replace('_', ' ')

        
            # Sort by importance
            sorted_idx = np.argsort(importances)
            sorted_importances = importances[sorted_idx]
            sorted_features = [clean_name(feature_names[i]) for i in sorted_idx]

            # Dark mode styling
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, len(sorted_features) * 0.3))
            ax.set_facecolor('#063672')  
            fig.patch.set_facecolor('#063672')

            # Draw lollipop chart
            ax.hlines(y=sorted_features, xmin=0, xmax=sorted_importances, color='#444', linewidth=1)
            ax.plot(sorted_importances, sorted_features, "o", markersize=10, color='#EF0107') 

            # Axes and labels
            ax.set_xlabel("Feature Importance", fontsize=12, color='white')
            ax.set_title("Feature Importances", fontsize=14, color='white', weight='bold')
            ax.tick_params(colors='white', labelsize=10)
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='white')
            fig.tight_layout()

            # Streamlit:
            st.pyplot(fig)

        # Define valid options (replace these with your actual values from your dataset if needed)
        positions = ['Defense', 'Midfield', 'Forward']

        possession_team = ['Arsenal', 'AFC Bournemouth', 'Aston Villa', 'Chelsea', 'Crystal Palace', 
                        'Everton', 'Leicester City', 'Liverpool', 'Manchester City', 'Manchester United', 
                        'Newcastle United', 'Norwich City', 'Southampton', 'Stoke City', 'Sunderland', 
                        'Swansea City', 'Tottenham Hotspur', 'Watford', 'West Bromwich Albion', 'West Ham United']


        play_patterns = ['From Corner',
        'From Counter',
        'From Free Kick',
        'From Throw In',
        'Regular Play',
        'From Goal Kick']

        # Streamlit widgets
        
        team = st.selectbox("🤩 _**Choose which team will be analyzed:**_", possession_team)
        for team_name in possession_team:
            if team == team_name:
                image = Image.open(f'../img_epl/{team}.jpg')
                st.image(image)
        position = st.selectbox("_**⚽ Choose the player role of who is trying to score:**_", positions)
        play_pattern = st.selectbox("_**↗️ Choose the type of play the scoring attempt is being made:**_", play_patterns)
        # Select period
        period = st.selectbox("_**⏱️ Choose either the 1st half or 2nd half of the match:**_", [1, 2], format_func=lambda x: f"First Half" if x == 1 else "Second Half")

        # Set valid minute range based on period
        if period == 1:
            minute = st.slider("_**⌛ Adjust to a specific minute in the first half:**_", 0.0, 53.0, 30.0)
        else:
            minute = st.slider("_**⌛ Adjust to a specific minute in the second half:**_", 45.0, 98.0, 70.0)

        x = st.slider("_**📍 Choose the horizontal position on the field where the scoring attempt is being made:**_", 60.0, 120.0, 100.0)
        y = st.slider("_**📍 Choose the vertical position on the field where the scoring attempt is being made:**_", 0.0, 80.0, 40.0)
        st.markdown("##### Look at how the player's position moves as you move the sliders")
        # Create Plotly figure
        fig = go.Figure()

        # Add player position
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(size=14, color='red'),
            name='Player Position',
            showlegend=False
        ))

        # Draw pitch lines manually
        pitch_shapes = []

        # Outer boundaries
        pitch_shapes.append(dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="white", width=3)))

        # Center line
        pitch_shapes.append(dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="white", width=3)))

        # Penalty areas
        pitch_shapes += [
            dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="white")),
            dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="white")),
        ]

        # 6-yard boxes
        pitch_shapes += [
            dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="white")),
            dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="white")),
        ]

        # Center circle (approximate)
        circle_points = 100
        circle_x = [60 + 10 * np.cos(2 * np.pi * i / circle_points) for i in range(circle_points)]
        circle_y = [40 + 10 * np.sin(2 * np.pi * i / circle_points) for i in range(circle_points)]
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='white'), showlegend=False))

        # Update layout
        fig.update_layout(
        title="🔴 Player Position",
        shapes=pitch_shapes,
        xaxis=dict(
            range=[0, 120],
            showgrid=False,
            zeroline=False,
            constrain='domain'  # ✨ Prevents excess padding on the x-axis
        ),
        yaxis=dict(
            range=[0, 80],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        height=430,
        margin=dict(l=0, r=0, t=40, b=0),  # ✨ Tighter margins
        plot_bgcolor='#1f4722'
    )
        
        # Show in Streamlit
        st.plotly_chart(fig)

        ## Request to API
        minute_in_half = int(minute) if period == 1 else int(minute - 45)

        input_data = {"match_period":period, "minute_in_half":minute_in_half, "possession_team":team, "play_pattern":play_pattern, "position":position, "x":x, "y":y}

        # Send request to FastAPI
        response = requests.post("http://127.0.0.1:8000/predict/goals/epl", json=input_data)
        # response = requests.post("http://localhost:8000/predict", json=input_data)

        if response.status_code == 200:
            result = response.json()
            probability = result['prediction']
            # st.markdown(f"### 🎯 Probability of Scoring: '{probability:.2%}'")
            st.markdown(f"### 🎯 Probability of Scoring: <span style='color:red'>{probability:.2%}</span>", unsafe_allow_html=True)
        else:
            st.error("Something went wrong")
            st.write(response.status_code)

    if league_selection == 'Bonus: Messi':
        image = Image.open('../img/campnou.webp')
        image_display.image(image)

        # Load trained model pipeline        
        model = joblib.load("messigoalsmodel_rf.pkl")

        st.markdown("###### The parameters below change the probability of scoring a goal. Adjust different parameters to see how the probability of scoring changes. Scoring probability is shown at the bottom of the page.")

        # Checkbox to show the importance of features used in this model
        important_features = st.checkbox("_Check this box if you would like to see how the parameters below influence the changes in the model (it will not affect the model's prediction if you click this):_", value=False)
        if important_features == True:
            ## Feature Importances
            numeric_features = ['match_period', 'minute_in_half', 'x', 'y']
            categorical_features = ['under_pressure','play_pattern']

            importances = model.named_steps['classifier'].feature_importances_
            feature_names = numeric_features + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features))

            # Clean feature names
            def clean_name(name):
                name = name.lower()
                if 'play_pattern_' in name:
                    name = name.replace('play_pattern_', '')
                return name.replace('_', ' ')

            # Sort by importance
            sorted_idx = np.argsort(importances)
            sorted_importances = importances[sorted_idx]
            sorted_features = [clean_name(feature_names[i]) for i in sorted_idx]

            # Dark mode styling
            plt.style.use('dark_background')
            fig, ax = plt.subplots(figsize=(10, len(sorted_features) * 0.3))
            ax.set_facecolor('#063672')  
            fig.patch.set_facecolor('#063672')

            # Draw lollipop chart
            ax.hlines(y=sorted_features, xmin=0, xmax=sorted_importances, color='#444', linewidth=1)
            ax.plot(sorted_importances, sorted_features, "o", markersize=10, color='#EF0107') 

            # Axes and labels
            ax.set_xlabel("Feature Importance", fontsize=12, color='white')
            ax.set_title("Feature Importances", fontsize=14, color='white', weight='bold')
            ax.tick_params(colors='white', labelsize=10)
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='white')
            fig.tight_layout()

            # Streamlit:
            st.pyplot(fig)

        image_messi = Image.open('../img/messi.jpg')
        st.image(image_messi)
        # Define valid options (replace these with your actual values from your dataset if needed)

        play_patterns = ['From Corner',
        'From Counter',
        'From Free Kick',
        'From Throw In',
        'Regular Play',
        'From Goal Kick']

        # Streamlit widgets
        play_pattern = st.selectbox("_**🥵 Choose the type of play the scoring attempt is being made:**_", play_patterns)
        
        under_pressure = st.checkbox("_**↗️ Choose if the player is under pressure when attempting to score a goal:**_", value=False)

        period = st.selectbox("_**⏱️ Choose either the 1st half or 2nd half of the match:**_", [1, 2], format_func=lambda x: f"1st Half" if x == 1 else "2nd Half")

        # Set valid minute range based on period
        if period == 1:
            minute = st.slider("_**⌛ Adjust to a specific minute in the first half:**_", 0.0, 53.0, 30.0)
        else:
            minute = st.slider("_**⌛ Adjust to a specific minute in the second half:**_", 45.0, 98.0, 70.0)

        x = st.slider("_**📍 Choose the horizontal position on the field where the scoring attempt is being made:**_", 60.0, 120.0, 100.0)
        y = st.slider("_**📍 Choose the vertical position on the field where the scoring attempt is being made:**_", 0.0, 80.0, 40.0)
        st.markdown("##### Look at how the player's position moves as you move the sliders")

        # Create Plotly figure
        fig = go.Figure()

        # Add player position
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers',
            marker=dict(size=14, color='red'),
            name='Player Position',
            showlegend=False
        ))

        # Draw pitch lines manually
        pitch_shapes = []

        # Outer boundaries
        pitch_shapes.append(dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="white", width=3)))

        # Center line
        pitch_shapes.append(dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="white", width=3)))

        # Penalty areas
        pitch_shapes += [
            dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="white")),
            dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color="white")),
        ]

        # 6-yard boxes
        pitch_shapes += [
            dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="white")),
            dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color="white")),
        ]

        # Center circle (approximate)
        circle_points = 100
        circle_x = [60 + 10 * np.cos(2 * np.pi * i / circle_points) for i in range(circle_points)]
        circle_y = [40 + 10 * np.sin(2 * np.pi * i / circle_points) for i in range(circle_points)]
        fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='white'), showlegend=False))

        # Update layout
        fig.update_layout(
        title="🔴 Player Position",
        shapes=pitch_shapes,
        xaxis=dict(
            range=[0, 120],
            showgrid=False,
            zeroline=False,
            constrain='domain'  # ✨ Prevents excess padding on the x-axis
        ),
        yaxis=dict(
            range=[0, 80],
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        height=430,
        margin=dict(l=0, r=0, t=40, b=0),  # ✨ Tighter margins
        plot_bgcolor='#1f4722'
    )
        
        # Show in Streamlit
        st.plotly_chart(fig)

        ## Request to API
        minute_in_half = int(minute) if period == 1 else int(minute - 45)

        input_data = {"match_period":period, "minute_in_half":minute_in_half, "play_pattern":play_pattern, "under_pressure":under_pressure, "x":x, "y":y}

        # Send request to FastAPI
        response = requests.post("http://127.0.0.1:8000/predict/goals/messi", json=input_data)

        if response.status_code == 200:
            result = response.json()
            probability = result['prediction']
            st.markdown(f"### 🎯 Probability of Scoring: <span style='color:red'>{probability:.2%}</span>", unsafe_allow_html=True)
        else:
            st.error("Something went wrong")
            st.write(response.status_code)

        # probability = model.predict_proba(input_df)[0][1]
        # st.markdown(f"### 🎯 Probability of Scoring: `{probability:.2%}`")

if option == 'Win Probability':
    # league_selection = st.selectbox("Choose a League or Bonus Item:", ['EPL', 'Bonus: Messi'])
    # if league_selection == 'EPL':
    image = Image.open('../img/epl.jpg')
    st.image(image)

    # Load trained model pipeline
    model = joblib.load("eplmatches5ymodel_rf.pkl")

    st.markdown("###### The parameters below change the probability of the home team winning the match. Adjust the parameters below to see how the probability changes. Winning probability is shown at the bottom of the page.")

    important_features = st.checkbox("_Check this box if you would like to see how the parameters below influence the changes in the model (it will not affect the model's prediction if you click this):_", value=False)
    if important_features == True:
        ## Feature Importances
        numeric_features = ['temperature_day', 'wind_speed',	'humidity',	'pressure',	'clouds']
        categorical_features = ['position_away', 'position_home', 'name', 'time_of_day']

        importances = model.named_steps['classifier'].feature_importances_
        feature_names = numeric_features + (model.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_features).tolist())
        
        # Clean feature names
        def clean_name(name):
            name = name.lower()
            if 'name_' in name:
                name = name.replace('name_', '')
            return name.replace('_', ' ')

     
        # Sort by importance
        sorted_idx = np.argsort(importances)
        sorted_importances = importances[sorted_idx]
        sorted_features = [clean_name(feature_names[i]) for i in sorted_idx]

        # Dark mode styling
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, len(sorted_features) * 0.3))
        ax.set_facecolor('#063672')  
        fig.patch.set_facecolor('#063672')

        # Draw lollipop chart
        ax.hlines(y=sorted_features, xmin=0, xmax=sorted_importances, color='#444', linewidth=1)
        ax.plot(sorted_importances, sorted_features, "o", markersize=10, color='#EF0107') 

        # Axes and labels
        ax.set_xlabel("Feature Importance", fontsize=12, color='white')
        ax.set_title("Feature Importances", fontsize=14, color='white', weight='bold')
        ax.tick_params(colors='white', labelsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.3, color='white')
        fig.tight_layout()

        # Streamlit:
        st.pyplot(fig)

    field_name = ['Emirates Stadium','Anfield', 'Bramall Lane', 'Brentford Community Stadium', 'Broadfield Stadium', 'Craven Cottage', 
                   'Elland Road', 'Etihad Stadium', 'Goodison Park', 'Kenilworth Road Stadium', 'King Power Stadium', 
                   'London Stadium', 'Molineux Stadium', 'Old Trafford', 'Portman Road Stadium', 'Selhurst Park', 
                   "St. James' Park", "St. Mary's Stadium", 'Stamford Bridge', 'The American Express Community Stadium', 
                   'The City Ground', 'Tottenham Hotspur Stadium', 'Turf Moor', 'Vicarage Road', 'Villa Park', 'Vitality Stadium']

    time_in_day = ['earlier', 'later']
    
    # Streamlit widgets
    
    # Select position
    pitch_name = st.selectbox("🏟️ _**Choose the pitch that the match is being played on (the pitch selected will match the name of the home_team):**_", field_name)
    for team_name in field_name:
        if pitch_name == team_name:
            image = Image.open(f'../img_eplpitch/{pitch_name}.jpg')
            st.image(image)

    home_position = st.selectbox("⬜ _**Choose the home team's current standing on the table:**_", np.arange(1,21,1).astype(float))
    away_position_list = [float(i) for i in range(1,21) if i != home_position] 
    away_position = st.selectbox("🟥 _**Choose the away team's current standing on the table:**_", away_position_list)

    day_temp = st.slider("🌡️ _**Choose the temperature during the day the match is played (Celsius):**_", -10.68, 33.06, 11.0)
    wind_speeds = st.slider("🌀 _**Choose the wind speeds during the day the match is played (meters/second):**_", 0.95, 20.12, 6.0)
    humidity_level = st.slider("🌫️ _**Choose the percentage of humidity reached that day:**_", 20.0, 100.00, 70.0)
    pressure_amount = st.slider("🥵 _**Choose the atmospheric pressure reached that day (millibars):**_", 964.0, 1043.0, 1014.0)
    cloudiness = st.slider("☁️ _**Choose the amout of cloud coverage seen that day:**_", 0.0, 100.0, 69.0)
    
    time = st.selectbox("⌛ _**Choose whether the match was played earlier in the day (before 2:30pm) or later in day (after 2:30pm):**_", time_in_day)

    input_data = {'position_away':away_position, 'position_home':home_position, 'temperature_day':day_temp, 'wind_speed':wind_speeds, 
                  'humidity':humidity_level, 'pressure':pressure_amount, 'clouds':cloudiness, 'name':pitch_name, 'time_of_day':time}

    # Send request to FastAPI
    response = requests.post("http://127.0.0.1:8000/predict/matchoutcome/epl", json=input_data)
    # response = requests.post("http://localhost:8000/predict", json=input_data)

    if response.status_code == 200:
        result = response.json()
        probability = result['prediction']
        st.markdown(f"### 🎯 Probability of Winning: <span style='color:red'>{probability:.2%}</span>", unsafe_allow_html=True)
    else:
        st.error("Something went wrong")
        st.write(response.status_code)
    

if option == 'Analysis Key Points':
    st.write("Coming Soon")
