import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



st.set_page_config(
    page_title="NYC Crime Accident",
    page_icon=":red_car:",
    layout="wide",
    initial_sidebar_state="expanded",
    
)

col1, col2 = st.columns([3, 1])
with col1:
    st.title("NYC Crime Accident")
with col2:
    st.image("nypd-patch.png", width=100)

st.markdown("""
This application is a Streamlit dashboard that can be used to analyze crime accidents in NYC 🗽💥🚗. 
The data includes information about accident location, date, time, contributing factors, and more. 
Use the sidebar menu to navigate between different sections of the app.
""")
def load_data():
    df = pd.read_csv('output_file.csv')
    df = df.rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
    return df

def show_raw_data(df):
    st.subheader('Raw Data')
    st.write(df)
def explore_data(df, subheader):
    st.subheader(subheader)
    
    st.write(df)

def describe_features(df):
    st.subheader('Features Description')
    features_description = pd.DataFrame({
        'Feature': df.columns,
        'Description': [
            'Randomly generated persistent ID for each complaint',
            'Exact date of occurrence for the reported event (or starting date of occurrence, if CMPLNT_TO_DT exists)',
            'Exact time of occurrence for the reported event (or starting time of occurrence, if CMPLNT_TO_TM exists)',
            'Ending date of occurrence for the reported event, if exact time of occurrence is unknown',
            'Ending time of occurrence for the reported event, if exact time of occurrence is unknown',
            'Description of offense corresponding with key code',
            'Indicator of whether crime was successfully completed or attempted, but failed or was interrupted prematurely',
            'The name of the borough in which the incident occurred',
            'Midblock Latitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)',
            'Midblock Longitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)',
            'Victim Age Group',
            'Victim Sex Description',
            'Exact hour of occurrence for the reported event',
            'Exact minute of occurrence for the reported event',
            'Exact second of occurrence for the reported event',
            'Time of occurrence bucketed into Morning, Afternoon or Evening',
            'Offense Description',
            'Crime Category'
        ]
    })
    st.write(features_description)

def visualize_data(df):
    st.subheader('Visualizing Data')
    
    st.write("### Distribution of Crimes by Borough")
    plt.figure(figsize=(12, 6))
    sns.countplot(x='BORO_NM', data=df, order=df['BORO_NM'].value_counts().index)
    st.pyplot(plt)
    
    st.write("### Distribution of Crimes by Victim Sex")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Victim Sex', data=df, order=df['Victim Sex'].value_counts().index)
    st.pyplot(plt)

    st.write("### Distribution of Crimes by Victim Age Group")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Victim Age Group', data=df, order=df['Victim Age Group'].value_counts().index)
    st.pyplot(plt)

    st.write("### Distribution of Crimes by Start Hour")
    plt.figure(figsize=(15, 6))
    sns.histplot(x='Start Hour', data=df, bins=24, kde=True)
    st.pyplot(plt)
    
    st.write("### Heatmap of Crimes by Day of Week and Hour")
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df['Day of Week'] = df['Start Date'].dt.dayofweek
    day_hour = df.groupby(['Day of Week', 'Start Hour']).size().reset_index(name='Count')
    day_hour = day_hour.pivot('Day of Week', 'Start Hour', 'Count')
    plt.figure(figsize=(15, 6))
    sns.heatmap(day_hour, cmap='coolwarm')
    st.pyplot(plt)

def draw_map(df):
    # st.subheader("Accident Locations")
    # df = df.dropna(subset=['Latitude', 'Longitude'])
    # st.map(df[["Latitude", "Longitude"]])
    st.subheader("Accident Locations")
    df = df.dropna(subset=['lat', 'lon'])
    st.map(df[["lat", "lon"]])

def preprocess_data(df):
    st.header("Add some preprocessing steps here")


def train_model():
    st.header("Add trainning steps here")

def predict():
     st.header("Add prediction steps here")
def main():
    df = load_data()

    # Create a selectboxed menu at sidebar
    menu = ['Data Exploration', 'Show Features', 'Feature Description', "Show Map", "Preprocessing", "Training", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Data Exploration':
        explore_data(df,subheader='Data Exploration')
        visualize_data(df)
    elif choice == 'Show Features':
        explore_data(df.columns, 'Show Features')
    elif choice == 'Feature Description':
        describe_features(df)
    elif choice == 'Show Map':
        draw_map(df)
    elif choice == 'Preprocessing':
        preprocess_data(df)
    elif choice == 'Training':
        train_model()
    elif choice == 'Prediction':
        predict()

        

if __name__ == '__main__':
    main()
