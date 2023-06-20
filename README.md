# Payment Behavior Analysis with Self-Organizing Maps (SOM)

This Python script utilizes a Self-Organizing Map (SOM) to categorize customers of a SaaS platform according to their workspace usage behavior. 

By grouping user's workspaces into clusters of similar usage patterns, I was able to target my sales efforts towards users that were most likely to uplift their payment plan.

## Context

A Self-Organizing Map (SOM) is a specific type of artificial neural network that utilizes unsupervised learning to transform complex, high-dimensional data into a simpler, often two-dimensional, representation.

## Process

1. **Data Loading and Preprocessing**: The script first loads workspace data and processes it by converting date fields to datetime, calculating 'DaysSinceFirstSeen', and creating an 'is_paying' column. Features are then scaled using sklearn's MinMaxScaler.

2. **Self-Organizing Map (SOM) Creation**: After preprocessing, the script creates a SOM using the processed data. This involves initializing and training the SOM, and visualizing it by plotting the SOM's distance map. It also keeps track of the number of paying and non-paying counts in each BMU.

3. **BMU Categorization**: Once the SOM is created, each workspace is assigned a BMU. Then, each BMU is categorized based on the percentage of paying workspaces it contains. This final dataframe, including workspace IDs, whether the workspace is paying or not, their BMUs, and BMU categories, is saved to `final_dataframe.csv`.

## Dependencies

The script requires the following Python libraries:

- pandas
- numpy
- matplotlib
- sklearn
- minisom

These dependencies can be installed using pip:

```sh
pip install pandas numpy matplotlib sklearn minisom
```

## Data

The script reads data from a CSV file, `som_rawdata_test.csv`, located in the `data/input` directory. This file should contain workspace data with specific features.

The script outputs the resulting dataframe, which includes assigned BMU categories, to a CSV file, `final_dataframe.csv`, in the `data/output` directory.

## Usage

You can run the script using the following command:

```sh
python platform_payment_som.py
```

# Understanding the Results

The visualization produced by this SOM script is a powerful tool for understanding high-dimensional data in two dimensions. Here, we will explain the key components of the visualization.

![image](https://github.com/skiss10/SOM-Customer-Analysis/assets/31713441/5472596d-28fd-420c-9228-27a7044023b4)

## Tiles
Each tile, or cell, in the grid represents a neuron in the Self-Organizing Map. Neurons are characterized by a weight vector of the same dimension as the input vectors (i.e., feature vectors from your dataset). The spatial position of a tile on the grid does not inherently hold a particular meaning; it is the relative distance between the tiles (neurons) that carries information. The neuron with the weight vector that is closest (in terms of Euclidean distance) to the input vector is designated as the Best Matching Unit (BMU) for that specific input vector. 

## Tile Color
The color of a tile represents the similarity of the high-dimensional data points mapped to that node: lighter tiles signify more similarity among mapped input vectors, while darker tiles indicate a higher level of diversity or dissimilarity among the corresponding data points. The color gradient provides a comparative visualization of regional clusters on the map, rather than an absolute measure of similarity or dissimilarity.

## Markers
The markers ('x' and 'o') are placed on all of the SOM's Best Matching Units to to denote whether the input data thats being mapping to a given winning neuron is either 'paying' ('o') or 'non-paying' ('x').

## Marker Size
The size of the markers indicates the logarithmic count of data points associated with that particular winning neuron (tile). A larger marker size implies that a higher number of input data points are mapped to that neuron in the SOM. Logarithmic scaling was used to reduce skewness in the visualization - This means that differences in counts are more visually discernible even when there's a large disparity in counts. 

By understanding these elements, we can analyze the output of the SOM more effectively, discern patterns in the data, and make more informed decisions based on those patterns.

In particular, I was looking for:

- Input data mapping to lighter color tiles (BMUs with similar input data).
- Tiles containing both x's and o's (indicating the presence of both paying and non-paying customers).
- Tiles with bigger o's than x's (indicating a higher number of paying customers compared to non-paying customers).

From there, I proceeded to target the non-paying customers within those light-tiled BMUs.

# Credits

This project was inspired by the Fraud Detection Self-Organizing Map (SOM) exercise in the [Deep Learning A-Z™: Hands-On Artificial Neural Networks](https://www.udemy.com/course/deeplearning/) course on Udemy. The course provides an extensive introduction to the field of deep learning and offers hands-on exercises that helped form the foundation for this project.

I want to express my gratitude to the course instructors for providing the knowledge and skills needed to initiate this project. If you're interested in deep learning, I highly recommend checking out this course.



