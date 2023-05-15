# NYC Crime Analysis Dashboard

NYC Crime Analysis Dashboard is a Streamlit app that allows you to explore and analyze crime data from New York City. It provides insights into crime incidents, patterns, and trends across different boroughs and neighborhoods.

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
- **Crime Trends**: Analyze crime trends over time and the top crimes.
- **Accident Locations Map**: Geographical visualization of accident locations, filterable based on Time Bucket and Crime Category.
- **Data Preprocessing**: Perform one-hot encoding on categorical columns and handle missing values.
- **Crime Clustering**: Perform clustering of crimes using different methods like KMeans, DBScan, and Gaussian Mixture Model.

## How to use

Run the following command in your terminal to start the Streamlit app:

```shell
streamlit run app.py
```

You can navigate through different features of the app using the sidebar menu.

## Data Preprocessing

The data preprocessing involves one-hot encoding of categorical columns and handling of missing values. You can choose which column to encode and whether to remove null values. 

## Contributing

If you want to contribute to this project and make it better, your help is very welcome. Create a pull request with your suggested changes.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

If you want to contact me you can reach me at `your_email@domain.com`.

## Project Status

This project is currently in development. More features will be added in the future.

---

*Please note: This is a basic template of README. You might want to add more details, adjust sections, or include other information such as screenshots, a list of contributors, or links to related projects or resources.*
Run the application:
```
cd "NYC Traffic Accidents"
streamlit run nyc_traffic_accident.py
```

Happy Traveling!
