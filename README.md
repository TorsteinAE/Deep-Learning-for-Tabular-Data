# Deep Learning for Tabular Data

# I. Introduction

Neural networks and deep learning are more popular than ever, but are mostly being used with images, audio, video, or natural language.
A lot of datasets come as tabular data.

Our goal is to explore when deep learning is a better choice than more standard, tree-based, machine learning models. 
To do this we will use several large tabular datasets, including the dataset used in the "Deep Learning for Coders..." book [Blue Book for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data), a dataset used to see who survived the [Titanic](https://www.kaggle.com/competitions/titanic/overview), a dataset to [Predict Student Performance from Game Play](https://www.kaggle.com/competitions/predict-student-performance-from-game-play).

We will use the Machine Learning Project Checklist from the "Hands-On ML..." book:
1. Frame the problem and look at the big picture
2. Get the data
3. Explore the data
4. Prepare the data
5. Explore models
6. Fine tune models
7. Present solution
8. Launch

To evaluate what is best we will take into consideration the predictive performance, training time, inference time and requirements, explainability for each of the datasets.

# II. Background

Tabular data is data that comes in the form of a table, like a .csv file. The rows represent an observation, and colums represent the features, or the attributes of the observation. We want to predict one feature, the dependent variable, based on the data of the other features, the independent variables.

Tabular data is one of the most common types of data that we see in many real-world applications. It is also usualy easy to understand the data, you can use statistical and mathematical opperations, and there are good practices in place to explore and get a lot of knowledge out of the data. Some things one has to consider when handling tabular data is how to handle missing values, ordinal columns, dates and other categorical variables.

According to "Deep Learning for Coders..." the vast majority of datasets can be modeled with:
* Decision tree ensambles, for structured data
* Deep learning, for unstructered data

It also recomends starting with endambles of decision trees, but says that when the dataset has some important high-cardinality categorical variables, or there are columns that would be best understood with neural networks, to also try deep learning.

Deep learning methods for tabular data via embeddings involve representing categorical variables as continuous vectors, or embeddings, which are then used as inputs to neural networks.
This approach has been shown to improve the performance of neural networks on tabular data.

# III. Data Collection and Preparation

For our experiments, we will be using several large tabular datasets:

## Titanic

This dataset contains information about the passengers on the Titanic, including whether they survived or not.
We want to create a model that predicts if a passenger survived or not, based on info like their name, age, sex...

## Blue Book for Bulldozers

This dataset contains information about the sale of used bulldozers, including their characteristics and sale prices.
We want to create a model that predicts the sales prize of bulldozers based on info like the sales date, the machineId...

## Cleaning and Preparation
We need to carry out some data cleaning and preparation procedures before we can use these datasets for modeling. Some typical actions we might use are:
* Handling missing values: To fill in the dataset's missing values, we may utilize methods like imputation.
* Managing ordinal columns: Ordinal columns are those whose values are ranked or have an order, such as "low," "medium," or "high." We might utilize methods like ordinal encoding or turn these columns into numerical values.
* Handling dates: We may extract useful features from dates such as year, month, and day, or convert them to a numerical representation such as the number of days since a certain date.
* Handling categorical variables: Categorical variables are variables with a limited number of possible values (e.g. "red", "green", "blue"). We may use techniques such as one-hot encoding or embedding to represent these variables numerically.
* Normalizing inputs: We may normalize the numerical input features to have a certain range, such as between (-1,1) or (0,1), to help with training and avoid numerical issues.

After these steps, we can use the cleaned and preprocessed data to train and evaluate our machine learning models, including deep learning models.

# IV. Experimental Design

We will use several different ML models for our testing

## Decision Tree

A decision tree is a type of supervised machine learning algorithm used for classification and regression analysis.
It is a tree-like model where each internal node represents a test on an attribute or feature, each branch represents the outcome of the test, and each leaf node represents a class label or a numerical value.

Explanation of the hyperparameters and training techniques used

## Random Forest

Random forest is a type of ensemble learning method that combines multiple decision trees to create a more accurate and stable model.
In a random forest, multiple decision trees are trained on different subsets of the training data and using different subsets of features.
The subsets of data and features are chosen randomly, which helps to reduce the correlation between the trees and prevent overfitting.

Explanation of the hyperparameters and training techniques used

## Deep Learning

The building blocks of deep learning models are artificial neural networks, which are inspired by the structure and function of the human brain.
Neural networks consist of layers of interconnected nodes (also called neurons), where each node applies a nonlinear function to its inputs and produces an output that is passed on to the next layer.

We will be using fast.ai tabular library to create a neural network for out datasets.

Explanation of the hyperparameters and training techniques used

## Evaluation

Evaluation metrics used in machine learning models typically include predictive performance, training time, inference time, and explainability: 
* Predictive performance is the most commonly used metric and refers to how well the model performs on the task it was designed for
* Training time measures how long it takes to train the model 
* Inference time measures how long it takes to make predictions once the model is trained
* Explainability refers to the degree to which the model can provide insight into its decision-making process

# V. Results

Presentation of the results for each model, including the evaluation metrics used

## Titanic Result
Comparison of the performance of the different models

## Blue Book for Bulldozers
Comparison of the performance of the different models

# VI. Discussion

Interpretation and discussion of the results

Comment what characteristics that may be better for tabular data, what is not, 

what type of architecture and hyperparamaters are best performing

Discussion of the limitations and challenges of using deep learning models for tabular data

# VII. Conclusion

Summary of the key findings of the project

Tabular data come in many forms.

How can we create a deep learning neural network and architecture best suited for the tabular dataset

What kind of preproccessing is crutial in deep learning for tabular data

Conclusion of when deep learning models are more suited than tree-based models for tabular data

Recommendations for future research in this area

# VIII. References

Howard, J. and Gugger, S. (2021) Deep learning for coders with FASTAI and pytorch: AI applications without a Phd.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sebastopol, CA: O'Reilly Media, Inc. 

Géron, A. (2021) Hands-on machine learning with scikit-learn and tensorflow: Concepts, tools, and techniques to build Intelligent Systems.
<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Beijing: O'Reilly. 
