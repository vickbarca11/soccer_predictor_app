import streamlit as st
from streamlit_navigation_bar import st_navbar
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
st.set_page_config(layout="centered", initial_sidebar_state='expanded')

page = st_navbar(['Home', 'Prediction Tools', 'References', 'About me'])

if page == "Home":
    st.title("Introduction")

    st.markdown("## How Predictions Work in Soccer")
    st.write("Soccer is a game of fine margins, where outcomes often hinge on split-second decisions. " \
    "By analyzing historical data—such as player positions, match context, and tactical patterns, machine learning models " \
    "can recognize trends and forecast likely outcomes, like whether a shot will result in a goal or a team will win that match. " \
    "These predictions provide powerful insights, helping analysts, coaches, and fans understand the game on a deeper level. " \
    "Whether it's improving strategy or helping you with your fantasy league, predictive models are transforming how we see and think about the beautiful game.")

    st.markdown("#### _**What Influences Soccer Match Predictions?**_")
    st.write("Soccer match predictions go beyond just team names, they're shaped by a mix of key factors like weather conditions, " \
    "venue dynamics, league standings, and recent performance. A home advantage, rainy weather, or a team's current form can all shift the odds. " \
    "By analyzing these elements together, predictive models provide a more accurate and insightful forecast of how a match might unfold. ")
    col1, col2, col3 = st.columns(3)
    with col1: 
        image_newcast_crys = Image.open('../img/epl_match_table_top.png')
        st.image(image_newcast_crys)

        st.write("_**The image below was captured at 1:30pm (EST) on April 16th, 1 hour before the start of the match. Google's probability outcome was " \
        "in favor of Newcastle winning at 58%**_") 
        
    with col2:
        image_weather = Image.open('../img/epl_match_hum_press.png')
        st.image(image_weather)
        image_table = Image.open('../img/epl_match_table_bottom.png')
        st.image(image_table)

        st.write("_**The stadium is Newcastle's home stadium, St. James' Park in London. Newcastle United is ranked 4th in the table, " \
        "while Crystal Palace is ranked 12th. Live weather data allows us to use these features for our model**_")
    
    with col3:
        image_match_tool = Image.open('../img/epl_match_app_top.png')
        st.image(image_match_tool)
        image_table = Image.open('../img/epl_match_app_bottom.png')
        st.image(image_table)

        st.write("_**Inputting all the features from the match, our application favors Newcastle winning at 54.18%**_")

    st.markdown("#### _**What Shapes Goal Scoring Predictions?**_")
    st.write("Predicting whether a shot will result in a goal isn't just guesswork,it's the art of analyzing key in-game details. " \
    "Factors like the player's role, shooting position on the field, the team they represent, and historical trends all play a part. " \
    "Even the minute of the match, whether the player is under pressure, and the type of play leading to the attempt can tip the scales. " \
    "By combining these insights, predictive models reveal the likelihood of a goal with surprising accuracy.")
    
    col1, col2 = st.columns([0.75,0.3])

    with col1:
        image_rice_goal = Image.open('../img/epl_match_ars_goal.png')
        st.image(image_rice_goal)
        st.write("_**This goal was scored by Arsenal's Declan Rice on March 9th against Manchester United " \
        "during a regular season match. The goal was scored between the 73rd and 74th minute during a regular play. " \
        "Adjusting the on our application, we see that the probability of scoring was about 54.36%.**_")

    with col2:
        image_app_top = Image.open('../img/epl_match_ars_goal_app_top.png')
        image_app_bottom = Image.open('../img/epl_match_ars_goal_app_bottom.png')
        st.image(image_app_top)
        st.image(image_app_bottom)
    
if page == 'Prediction Tools':
    
    st.sidebar.markdown("#### 🔩 Select your prediction tool:")
    option = st.sidebar.selectbox("", ['Match Outcome', 'Scoring Positions'])
    if option == 'Match Outcome':
        # Sticky container style and empty slot for EPL
        st.markdown("""
        <style>
            .sticky-prob {
                position: fixed;
                top: 80px;
                right: 30px;
                width: 280px;
                background-color: #063672; /* Updated background */
                color: white; /* Text color */
                padding: 15px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
                z-index: 9999;
                text-align: center;
                font-family: 'Segoe UI', sans-serif;
            }
        </style>
    """, unsafe_allow_html=True)
        
        epl_match_prob_container = st.empty()
        image = Image.open('../img/epl.jpg')
        st.image(image)

        # Load trained model pipeline
        model = joblib.load("eplmatches5ymodel_rf.pkl")

        st.markdown("###### Adjust the parameters below to see how the features affect the probability of the home team winning.")

        important_features = st.checkbox("_Check this box to see the amout of influence the features have on the model "
        "(it will not affect the model's prediction if you click here):_", value=False, key="win_prob_feature_importance")

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
        pitch_name = st.selectbox("🏟️ _**Choose the home pitch for the match (the pitch selected who the home_team is):**_", field_name)
        for team_name in field_name:
            if pitch_name == team_name:
                image = Image.open(f'../img_eplpitch/{pitch_name}.jpg')
                st.image(image)

        home_position = st.selectbox("⬜ _**Choose the home team's current standing on the table:**_", np.arange(1,21,1).astype(float))
        away_position_list = [float(i) for i in range(1,21) if i != home_position] 
        away_position = st.selectbox("🟥 _**Choose the away team's current standing on the table:**_", away_position_list)

        day_temp = st.slider("🌡️ _**Choose temperature at the start of the match (Celsius):**_", -10.68, 33.06, 11.0)
        wind_speeds = st.slider("🌀 _**Choose the wind speed at the start of the match (meters/second):**_", 0.95, 20.12, 6.0)
        humidity_level = st.slider("🌫️ _**Choose the humidity percentage at the start of the match:**_", 20.0, 100.00, 70.0)
        pressure_amount = st.slider("🥵 _**Choose the atmospheric pressure at the start of the match (millibars):**_", 964.0, 1043.0, 1014.0)
        cloudiness = st.slider("☁️ _**Choose the percentage of cloud coverage at the start of the match:**_", 0.0, 100.0, 69.0)
        
        time = st.selectbox("⌛ _**Choose if the match was played earlier in the day (before or at 2:30pm) or later in day (after 2:30pm):**_", time_in_day)

        input_data = {'position_away':away_position, 'position_home':home_position, 'temperature_day':day_temp, 'wind_speed':wind_speeds, 
                    'humidity':humidity_level, 'pressure':pressure_amount, 'clouds':cloudiness, 'name':pitch_name, 'time_of_day':time}

        # Send request to FastAPI
        response = requests.post("http://127.0.0.1:8000/predict/matchoutcome/epl", json=input_data)
        # response = requests.post("http://localhost:8000/predict", json=input_data)

        if response.status_code == 200:
            result = response.json()
            probability = result['prediction']
            # st.markdown(f"### 🎯 Probability of Winning: <span style='color:red'>{probability:.2%}</span>", unsafe_allow_html=True)
        
            epl_match_prob_container.markdown(
            f"<div class='sticky-prob'>⚽ <strong>EPL - Winning Probability:</strong> "
            f"<span style='color:#EF0107; font-size: 1.5em'>{probability:.2%}</span></div>",
            unsafe_allow_html=True
        )
        else:
            st.error("Something went wrong")
            st.write(response.status_code)

    if option == "Scoring Positions":
        image_display = st.empty()

        tab1, tab2, tab3 = st.tabs(["English Premier League", "La Liga", "Messi"])

        # Create a container to dynamically update the probability
        prob_container = st.empty()
        with tab1:
            # Sticky container style and empty slot for EPL
            st.markdown("""
            <style>
                .sticky-prob {
                    position: fixed;
                    top: 80px;
                    right: 30px;
                    width: 280px;
                    background-color: #063672; /* Updated background */
                    color: white; /* Text color */
                    padding: 15px;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
                    z-index: 9999;
                    text-align: center;
                    font-family: 'Segoe UI', sans-serif;
                }
            </style>
        """, unsafe_allow_html=True)

            epl_prob_container = st.empty()

            image_epl = Image.open('../img/epl.jpg')
            image_display.image(image_epl)

            # Load trained model pipeline
            model = joblib.load("eplgoalsmodel_rf.pkl")
            
            st.markdown("###### Adjust the parameters below to see how the features affect the probability of scoring.")
            
            ## Checking this box allows the user to see the feature importances
            important_features = st.checkbox("_Check here to see the amount of influence the features have on the model "
            "(it will not affect the model's prediction if you click here):_", value=False, key="epl_goal_feature_importance")

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
            
            team = st.selectbox("🤩 _**Choose which team will be used for the model:**_", possession_team)
            for team_name in possession_team:
                if team == team_name:
                    image_epl_team = Image.open(f'../img_epl/{team}.jpg')
                    st.image(image_epl_team)
            position = st.selectbox("_**⚽ Choose the player role of the goal scorer:**_", positions)
            play_pattern = st.selectbox("_**↗️ Choose in what way the goal is being scored:**_", play_patterns)
            # Select period
            period = st.selectbox("_**⏱️ Choose either the 1st half or 2nd half of the match:**_", [1, 2], format_func=lambda x: f"First Half" if x == 1 else "Second Half")

            # Set valid minute range based on period
            if period == 1:
                minute = st.slider("_**⌛ Adjust to a specific minute in the first half:**_", 0.0, 53.0, 30.0, key="first_half_minute")
            else:
                minute = st.slider("_**⌛ Adjust to a specific minute in the second half:**_", 45.0, 98.0, 70.0, key="second_half_minute")

            x = st.slider("_**📍 Choose the player's horizontal position on the field:**_", 60.0, 120.0, 100.0, key='x_position')
            y = st.slider("_**📍 Choose the player's vertical position on the field:**_", 0.0, 80.0, 40.0, key='y_position')
            st.markdown("###### Look at how the player's position moves as you move the sliders above")
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

                epl_prob_container.markdown(
                f"<div class='sticky-prob'>⚽ <strong>EPL - Scoring Probability:</strong> "
                f"<span style='color:#EF0107; font-size: 1.5em'>{probability:.2%}</span></div>",
                unsafe_allow_html=True
            )
            else:
                st.error("Something went wrong")
                st.write(response.status_code)

        with tab2:
            st.write("Coming Soon:")

        with tab3:
            
            # Sticky container style and empty slot for Messi
            st.markdown("""
            <style>
                .sticky-prob {
                    position: fixed;
                    top: 80px;
                    right: 30px;
                    width: 280px;
                    background-color: #063672;  /* Desired dark blue */
                    color: white;  /* Text color */
                    padding: 15px;
                    border-radius: 20px;  /* Increased to make the box rounder */
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
                    z-index: 9999;
                    text-align: center;
                    font-family: 'JetBrains Mono', monospace;
                }

                .sticky-prob strong {
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)
            messi_prob_container = st.empty()

            image_campnou = Image.open('../img/campnou.webp')
            image_display.image(image_campnou)

            # Load trained model pipeline        
            model = joblib.load("messigoalsmodel_rf.pkl")

            st.markdown("###### Adjust the parameters below to see how the features affect the probability of scoring.")

            # Checkbox to show the importance of features used in this model
            important_features = st.checkbox("_Check here to see the amout of influence the features have on the model "
            "(it will not affect the model's prediction if you click here):_", value=False, key="messi_goal_feature_importance")

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
            play_pattern = st.selectbox("_**↖️ Choose in what way the goal is being scored:**_", play_patterns)
            
            under_pressure = st.checkbox("_**🥵 Choose if the player is under pressure when attempting to score a goal:**_", value=False)

            period = st.selectbox("_**⏱️ Choose either the 1st half or 2nd half of the match:**_", [1, 2], format_func=lambda x: f"1st Half" if x == 1 else "2nd Half")

            # Set valid minute range based on period
            if period == 1:
                minute = st.slider("_**⌛ Adjust to a specific minute in the first half:**_", 0.0, 53.0, 30.0)
            else:
                minute = st.slider("_**⌛ Adjust to a specific minute in the second half:**_", 45.0, 98.0, 70.0)

            x = st.slider("_**📍 Choose the player's horizontal position on the field:**_", 60.0, 120.0, 100.0)
            y = st.slider("_**📍 Choose the player's vertical position on the field:**_", 0.0, 80.0, 40.0)
            st.markdown("###### Look at how the player's position moves as you move the sliders above")

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

                messi_prob_container.markdown(
                f"<div class='sticky-prob'>⚽ <strong>Messi - Scoring Probability:</strong> "
                f"<span style='color:red; font-size: 1.5em'>{probability:.2%}</span></div>",
                unsafe_allow_html=True
            )
            else:
                st.error("Something went wrong")
                st.write(response.status_code)

if page == "References":
    doc_page = st.sidebar.radio("**Go to**", ["GitHub", "SportMonks", "Statsbomb"])
    if doc_page == "GitHub":
        st.sidebar.link_button("🚀 GitHub Repo", url="https://github.com/vickbarca11/soccer_predictor_app", type="primary")
    elif doc_page == "SportMonks":
        st.sidebar.link_button("🚀 SportMonks", url="https://www.sportmonks.com/", type="primary")
        st.markdown("A great beginner tutorial for using the SportMonks API with python:")
        st.link_button("Beginner's Guide with Python:", url="https://www.sportmonks.com/blogs/a-developers-playbook-harnessing-the-power-of-sportmonks-football-api-with-python/", type="primary")
        st.markdown("For SportMonks' Football API documentation, you can click on the link below:")
        st.link_button("SportMonks Football API Documentation", url="https://docs.sportmonks.com/football", type="primary")
    elif doc_page == "Statsbomb":
        st.sidebar.link_button("🚀 Statsbomb", url="https://statsbomb.com/", type="primary")
        st.markdown("For the documentation on the Statsbomb available open datasets, you can visit the link below." \
        " This is the dataset utilized for this web application to create our goal scoring predictive models" \
        " for the Premier League 2015/2016 season, as well as Messi's career at FC Barcelona:")
        st.link_button("Statsbombpy Documentation", url="https://github.com/statsbomb/statsbombpy", type="primary")

if page == "About me":
    site_page = st.sidebar.radio("**Go to**", ["Bio", "LinkedIn"])
    if site_page == "Bio":
        st.write("Hi, I'm Victor Chang—a data analyst, lifelong soccer fan, and builder of tools that turn stats into strategy. " \
        "With a background in mechanical engineering and a current role in the U.S. Army, I've honed a skill set grounded in " \
        "precision, discipline, and data-driven problem-solving. From optimizing logistics for 200+ personnel to driving " \
        "20% cost savings through smart analysis, I've consistently used data to deliver impact where it matters most. " \
        "Now, as I pursue my MBA, I'm focused on blending business insights with technical expertise to create real-world solutions. " \
        "This interactive soccer analytics web app was designed to uncover what drives goals and match outcomes in the Premier League. " \
        "Using datasets from SportMonks and StatsBombPy, the app featured two predictive models that let users tweak key variables—like " \
        "shot location, match context, or player stats—to explore how they influence goal probability and match results. " \
        "With a clean UI and intuitive controls, it was built for both casual fans looking to explore the game in a new light and " \
        "professionals seeking deeper strategic insights. Whether you're a club, coach, or just a football fanatic, my goal is to " \
        "turn raw numbers into smarter play—and more wins. Let's connect on LinkedIn and talk football, data, or who is going to win the World Cup.")

    elif site_page == "LinkedIn":
        st.sidebar.markdown(
            """
            <style>
            .linkedin-button {
                background-color: #0077B5;
                color: white;
                padding: 0.3em 1em;
                border: none;
                border-radius: 10px;
                text-align: center;
                text-decoration: none;
                display: block;
                font-size: 16px;
                font-weight: bold;
                margin-top: 10px;
                cursor: pointer;
            }
            .linkedin-button:hover {
                background-color: #005983;
            }
            </style>
            <a href="https://www.linkedin.com/in/victorchang1993/" target="_blank">
                <div class="linkedin-button">🔗 Victor's LinkedIn</div>
            </a>
            """,
            unsafe_allow_html=True
        )

