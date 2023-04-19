# Deep Learning for Tabular Data

# I. Introduction

Neural networks and deep learning are more popular than ever, but are mostly being used with images, audio, video, or natural language.
A lot of datasets come as tabular data.
Our goal is to explore when deep learning is a better choice than more standard, tree-based, machine learning models. 
To do this we will use several large tabular datasets, including the dataset used in the "Deep Learning for Coders..." book [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data), a dataset used to see who survived the [Titanic](https://www.kaggle.com/competitions/titanic/overview), a dataset to [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play), dataset4, dataset 5, ... 
To evaluate what is best we will take into consideration the predictive performance, training time, inference time and requirements, explainability for each of the datasets.

# II. Background

Tabular data is data that comes in the form of a table, like a .csv file. The rows represent an observation, and colums represent the features, or the attributes of the observation. We want to predict one feature, the dependent variable, based on the data of the other features, the independent variables.

Tabular data is one of the most common types of data that we see in many real-world applications. It is also usualy easy to understand the data, you can use statistical and mathematical opperations, and there are good practices in place to explore and get a lot of knowledge out of the data. Some considerations one have to consider when handling tabular data is how to handle missing values, ordinal columns, dates and other categorical variables.

According to "Deep Learning for Coders..." the vast majority of datasets can be modeled with:
* Decision tree ensambles, for structured data
* Deep learning, for unstructered data

If we were to go a non deep learning route, one would use decision trees or random forests.

Overview of deep learning methods for tabular data via embeddings

# III. Data Collection and Preparation

Description of the datasets used in the experiments

Titanic inneholder informasjon om passasjerene, inkludert om de overlevde eller ikke

Bulldozer

Students

Explanation of the data cleaning and preprocessing steps taken

Missing values

Ordinal Columns

Dates

Categorical

Normalize between -1,1 or 0,1

# IV. Experimental Design

Overview of the models used (tree-based models and neural network-based models with embeddings) In addition, one can consider including other libraries for tree-based models LightGBM and XGBoost.

Explanation of the hyperparameters and training techniques used for each model

Description of the evaluation metrics used (predictive performance, training time, inference time, explainability)

# V. Results

Presentation of the results for each model, including the evaluation metrics used

## Blue Book for Bulldozers
Comparison of the performance of the different models

## Titanic Result
Comparison of the performance of the different models

# VI. Discussion

Interpretation of the results

Ser etter en spesiell relevant karakterestikk i datasetene. 

Explanation of when deep learning models are more suited than tree-based models for tabular data

Discussion of the limitations and challenges of using deep learning models for tabular data

# VII. Conclusion

Summary of the key findings of the project
We found that when a tabular dataset is ... it is better to use deep learning

Recommendations for future research in this area

# VIII. References

Howard, J. and Gugger, S. (2021) Deep learning for coders with FASTAI and pytorch: AI applications without a Phd.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sebastopol, CA: O'Reilly Media, Inc. 
