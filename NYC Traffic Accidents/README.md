# NYC Crime Analysis Dashboard

NYC Crime Analysis Dashboard is a Streamlit app that allows you to explore and analyze crime data from New York City. It provides insights into crime incidents, patterns, and trends across different boroughs and neighborhoods. By feeding the hidtorical crime data into our Machine Learning models, users will have a clear map view of where are the crimes geographically clustered in NYC. This will potentially provide users with some guidances when planning their trips or housing.

The layout has sidebar with options to filter the dataset
- Data Exploration
- Show Features
- Feature Description
- Show Map
- Preprocessing
- Show Cluster

## Installation

The required Python libraries for this project are provided in the `requirements.txt` file, which can be installed using pip:

```shell
pip install -r requirements.txt
```

## Features

- **Data Exploration**: View raw data and different visualizations like various distributions and a heatmap of crimes.
- **Feature Description**: Detailed description of each feature in the dataset.
- **Crime Trends**: Analyze crime trends over time and list out top crimes.
- **Accident Locations Map**: Geographical visualization of accident locations, filterable based on Time Bucket and Crime Category.
- **Data Preprocessing**: Perform one-hot encoding on categorical columns and handle missing values.
- **Crime Clustering**: Perform clustering of crimes using different methods, including KMeans, DBScan, and Gaussian Mixture Model. Moreover, display the cluster centroids in a map view and comparison of model performance.

## How to use

Run the following command in your terminal to start the Streamlit app:

```
cd "NYC Traffic Accidents"
streamlit run nyc_traffic_accident.py
```

You can navigate through different features of the app using the sidebar menu.

## Data Source

NYPD_Complaint_Data_Historic (NYC Open Data)
Crime_category (Self defined mapping)

## Data Cleaning

Before feeding the data into this Web App, we cleaned it by removing missing values, mapping crimes into self defined categories (e.g., sexual crime, violence, property crime, white collar crime, etc.), and add Time Bucket feature (e.g., morning, afternoon, evening, late night) 

## Model Selection
Users can choose from the following 3 models to train crime clustering:

- K-means: it is the most commonly used clustering algorithm
- Gaussion Mixture Model: comparing to K-means, which determine "hard" clustering, the GMM model has softer clusteing as it assigns probabilities to each data point based on how likely it belongs to different clusters (i.e., it does not "assign" each datapoint to each cluster), and it allows overlapping clusters.
- DBSCAN: this is a density based clustering algorithm, and it automatically learns the optimal number of clusters (instead of defined by hyperparameter). It is usally used for higher dimensioanl clustering, and could run really slow on large dataset. Note that when training the DBSCAN in this Web App, you need to select a specific crime type (e.g., sexual crime) to train the model, otherwise it could take hours to run.

## Evauation
- BIC score: we use BIC score to find the optimal number of clusters for K-means and GMM
- Davies bouldin score: it measures the similarity among different clusters, thus lower score indicate better performance
- Calinski harabasz score: ratio of between-class dispersion versus in-class dispersion, thus the higher score indicate better performance

we use DB score and CH score to find the best performing model, since they show consistence result (Kmeans>GMM>DBSCAN), we are only displaying DB score in the Web App.

## Contributing

If you want to contribute to this project and make it better, your help is very welcome. Create a pull request with your suggested changes.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

If you want to contact me you can reach me at `jc2888@cornell.edu`.

## Project Status

This project is currently in development. More features will be added in the future.
\

Happy Traveling!
