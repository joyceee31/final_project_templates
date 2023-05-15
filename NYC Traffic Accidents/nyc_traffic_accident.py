import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
import plotly.express as px
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from streamlit_folium import folium_static, st_folium
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import mode


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
Welcome to the NYC Crime Analysis Dashboard 🗽🔒🚨! 
This powerful tool allows you to explore and analyze crime data from the vibrant city of New York. 
Gain valuable insights into crime incidents, patterns, and trends across different boroughs and neighborhoods. 
 Let's dive into the data and uncover the stories behind NYC's crime landscape!
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
    print(list(df.columns))

    st.write(df)


def describe_features(df):
    st.subheader('Features Description')

    features_description = pd.DataFrame({
        'Feature': ['Complaint ID', 'Start Date', 'Start Time', 'Offense Description',
                    'Completion Status', 'BORO_NM', 'Latitude', 'Longitude',
                    'Victim Age Group', 'Victim Sex', 'Start Hour', 'Time Bucket',
                    'Crime_Category'],
        'Description': [
            'Unique identifier for each complaint',
            'Exact date of occurrence for the reported event (or starting date of occurrence, if end date exists)',
            'Exact time of occurrence for the reported event (or starting time of occurrence, if end time exists)',
            'Description of offense',
            'Indicator of whether crime was successfully completed or attempted, but failed or was interrupted prematurely',
            'The name of the borough in which the incident occurred',
            'Midblock Latitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)',
            'Midblock Longitude coordinate for Global Coordinate System, WGS 1984, decimal degrees (EPSG 4326)',
            'Victim Age Group',
            'Victim Sex Description',
            'Exact hour of occurrence for the reported event',
            'Time of occurrence bucketed into Morning, Afternoon or Evening',
            'Crime Category'
        ]
    })
    st.write(features_description)


def visualize_data(df):
    st.subheader('Visualizing Data')

    def plot_crimes_by_borough():
        st.write("### Distribution of Crimes by Borough")
        plt.figure(figsize=(12, 6))
        sns.countplot(x='BORO_NM', data=df,
                      order=df['BORO_NM'].value_counts().index)
        st.pyplot(plt)

    def plot_crimes_by_sex():
        st.write("### Distribution of Crimes by Victim Sex")
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Victim Sex', data=df,
                      order=df['Victim Sex'].value_counts().index)
        st.pyplot(plt)

    def plot_crimes_by_age_group():
        st.write("### Distribution of Crimes by Victim Age Group")
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Victim Age Group', data=df,
                      order=df['Victim Age Group'].value_counts().index)
        st.pyplot(plt)

    def plot_crimes_by_hour():
        st.write("### Distribution of Crimes by Start Hour")
        plt.figure(figsize=(15, 6))
        sns.histplot(x='Start Hour', data=df, bins=24, kde=True)
        st.pyplot(plt)

    def plot_heatmap_day_hour():
        st.write("### Heatmap of Crimes by Day of Week and Hour")
        df['Start Date'] = pd.to_datetime(df['Start Date'])
        df['Day of Week'] = df['Start Date'].dt.dayofweek
        day_hour = df.groupby(['Day of Week', 'Start Hour']
                              ).size().reset_index(name='Count')
        day_hour = day_hour.pivot('Day of Week', 'Start Hour', 'Count')
        plt.figure(figsize=(15, 6))
        sns.heatmap(day_hour, cmap='coolwarm')
        st.pyplot(plt)

    plot_dict = {
        'Distribution of Crimes by Borough': plot_crimes_by_borough,
        'Distribution of Crimes by Victim Sex': plot_crimes_by_sex,
        'Distribution of Crimes by Victim Age Group': plot_crimes_by_age_group,
        'Distribution of Crimes by Start Hour': plot_crimes_by_hour,
        'Heatmap of Crimes by Day of Week and Hour': plot_heatmap_day_hour
    }

    plot_name = st.selectbox("Choose a plot", list(plot_dict.keys()))
    plot_dict[plot_name]()


###################### FETCH DATASET #######################

def show_clusters(df):
    st.header("Crime Clusters in Manhattan")

    df['Start Date'] = pd.to_datetime(df['Start Date'], format='%m/%d/%Y')

    # Filter the dataset to include only data points from the year 2022
    df = df[df['Start Date'].dt.year == 2022]

    # Create a list of values to remove
    values_to_remove = ['4. White-Collar Crimes', '5. Drug-Related Crimes',
                        '6. Traffic Offenses', '8. Miscellaneous Crimes']

    # Filter the dataset by excluding the specified values fromb the 'Crime_Category' column
    df = df[~df['Crime_Category'].isin(values_to_remove)]

    scope_boro = ['MANHATTAN']
    df = df[df['BORO_NM'].isin(scope_boro)]

    df_crime = df[['lat', 'lon', 'Crime_Category']]

    df_crime_geo = df_crime[['lat', 'lon']]
    st.header("Davies Bouldin Score")
    kmean = KMeans(n_clusters=20)
    kmean = kmean.fit(df_crime_geo)
    k_labels = kmean.predict(df_crime_geo)
    k_score = davies_bouldin_score(df_crime_geo, k_labels)
    gmm = GaussianMixture(n_components=13).fit(df_crime_geo)
    g_labels = gmm.predict(df_crime_geo)
    g_score = davies_bouldin_score(df_crime_geo, g_labels)

    # Create a dataframe with the scores
    scores_df = pd.DataFrame({
        'KMeans': [k_score],
        'Mixture Model': [g_score],
        'DBScan': [0.8546630132415487]
    }, index=['DB Score'])

    # Display the table
    st.table(scores_df)

    clustering_method = st.selectbox("Choose a clustering method", [
                                     "KMeans", "Gaussian Mixture Model", "DBScan"])
    # If DBScan is selected, provide option to choose a specific crime category
    if clustering_method == "DBScan":
        crime_category = st.selectbox("Choose a crime category for DBScan",
                                      ["1. Violent Crimes", "2. Sexual Crimes", "3. Property Crimes"])
        df = df[df['Crime_Category'] == crime_category]

    if clustering_method == "KMeans":
        model = KMeans(n_clusters=20)
    elif clustering_method == "Gaussian Mixture Model":
        model = GaussianMixture(n_components=13)
    elif clustering_method == "DBScan":
        scaler = StandardScaler()
        X = scaler.fit_transform(df_crime[['lat', 'lon']])
        model = DBSCAN(eps=0.5, min_samples=5)
        df_crime['Cluster'] = model.fit_predict(X)
    else:
        df_crime['Cluster'] = model.fit_predict(df_crime[['lat', 'lon']])

    df_cluster_centers = df_crime.groupby('Cluster').agg(
        {'lat': 'mean', 'lon': 'mean', 'Crime_Category': lambda x: mode(x)[0][0]}).reset_index()

    m = folium.Map(location=[df_crime['lat'].mean(),
                   df_crime['lon'].mean()], zoom_start=13)

    for idx, row in df_cluster_centers.iterrows():
        folium.Marker(location=[row['lat'], row['lon']],
                      popup=row['Crime_Category'],
                      icon=folium.Icon(color='blue')).add_to(m)

    # display the map in Streamlit
    folium_static(m)


def draw_map(df):
    st.subheader("Accident Locations")

    # Define options for the selectboxes
    time_buckets = df['Time Bucket'].unique().tolist()
    crime_categories = df['Crime_Category'].unique().tolist()

    # Create the selectboxes
    time_bucket_filter = st.selectbox(
        'Filter map by Time Bucket', time_buckets)
    crime_category_filter = st.selectbox(
        'Filter map by Crime Category', crime_categories)

    # Filter data based on selected options
    df = df[(df['Time Bucket'] == time_bucket_filter) & (
        df['Crime_Category'] == crime_category_filter)]

    # Remove rows with missing lat/lon data
    df = df.dropna(subset=['lat', 'lon'])

    # Draw the map
    st.map(df[["lat", "lon"]])


def preprocess_data(df):
    st.header("Preprocessing Data")

    categorical_columns = ['BORO_NM', 'Victim Sex']
    st.write("### Select a categorical column to encode")
    column_to_encode = st.selectbox("Choose a column", categorical_columns)

    # Perform One-Hot encoding on the selected column
    df_encoded = pd.get_dummies(
        df, columns=[column_to_encode], prefix=column_to_encode)

    # Show the updated DataFrame with encoded columns
    st.write(df_encoded)
    return df_encoded


def train_model(df):
    st.header("Training the Model")
    feature_columns = [col for col in df.columns if col.startswith(
        'BORO_NM_') or col.startswith('Victim Sex_') or col == 'Start Hour']
    X = df[feature_columns]
    X = df[['BORO_NM_encoded', 'Victim Sex_encoded', 'Start Hour']]
    y = df['Crime_Category_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    st.write("Model trained successfully")

    return model


def crime_trends(df):
    st.subheader('Crime Trends Over Time')
    df['Start Date'] = pd.to_datetime(df['Start Date'])
    df_trends = df.groupby(df['Start Date'].dt.year)[
        'Crime_Category'].count().reset_index()
    fig = px.line(df_trends, x='Start Date', y='Crime_Category',
                  title='Yearly Crime Trends')
    st.plotly_chart(fig)

    st.subheader('Top {} Crimes'.format(10))
    top_n = df['Offense Description'].value_counts()[:n].reset_index()
    top_n.columns = ['Offense Description', 'Count']
    fig = px.bar(top_n, x='Offense Description', y='Count',
                 title='Top {} Crimes'.format(10))
    st.plotly_chart(fig)


def predict(model, df):
    st.header("Making Predictions")
    X = df[['BORO_NM_encoded', 'Victim Sex_encoded', 'Start Hour']]

    y_pred = model.predict(X)

    df['Predicted Crime Category'] = y_pred
    st.write(df.head())


def draw_clusters_map(df):
    st.subheader("Map of Accidents Clusters in Manhattan")

    # Create a map centered around Manhattan
    map_data = folium.Map(location=[40.7831, -73.9712], zoom_start=11)

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat))

    # Create a color dictionary for clusters
    colors = plt.cm.rainbow(np.linspace(0, 1, len(df['y'].unique())))

    color_dict = {}
    for idx, label in enumerate(df['y'].unique()):
        color_dict[label] = colors[idx]

    # Add points to the map
    for lat, lon, label in zip(df['lat'], df['lon'], df['y']):
        folium.CircleMarker(
            [lat, lon],
            radius=5,
            color=matplotlib.colors.rgb2hex(color_dict[label]),
            fill=True,
            fill_color=matplotlib.colors.rgb2hex(color_dict[label]),
            fill_opacity=0.6
        ).add_to(map_data)

    # Display the map in the app
    folium_static(map_data)


def main():
    df = load_data()

    # Create a selectboxed menu at sidebar
    menu = ['Data Exploration', 'Feature Description', 'Crime Trends',
            'Show Clusters', "Show Map", "Preprocessing", "Training", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Data Exploration':
        st.subheader('Data Exploration')
        st.markdown("""
        Welcome to the data exploration section. Here you can have a look at the raw data as well as some visualizations that help understand the dataset better.
        You can select different visuals from the dropdown menu below. The options include various distributions and a heatmap of crimes.
        """)
        explore_data(df, subheader='Raw Data')
        visualize_data(df)
    elif choice == 'Feature Description':
        describe_features(df)
    elif choice == 'Show Map':
        st.subheader('Accident Locations Map')
        st.markdown("""
        This section displays a geographical visualization of the accidents. The map below shows locations of the accidents.
        You can filter the locations displayed on the map based on Time Bucket and Crime Category using the dropdown menus.
        """)
        draw_map(df)
    elif choice == 'Preprocessing':
        preprocess_data(df)
    elif choice == 'Training':
        st.markdown("""
        The dataset used for this project has the following specifics:
        
        • **Borough**: We will focus solely on Manhattan, disregarding the other boroughs.
        
        • **Year**: We will only consider data from the year 2022, narrowing down the range from 2021-2022.
        
        • **Crime Category**: We will concentrate on three categories—Violent Crimes, Sexual Crimes, and Property Crimes—selected from a total of eight categories.
        
        By implementing these filters, we were able to reduce the dataset size to 102,032 data points.
        """)
        show_clusters(df)
    # elif choice == 'Training':
    #     train_model(df)
    #     draw_clusters_map(df)  # Add this line
    elif choice == 'Crime Trends':
        crime_trends(df)
    elif choice == 'Prediction':
        predict()


if __name__ == '__main__':
    main()
