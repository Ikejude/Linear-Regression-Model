# Import libraries!
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json
import pickle
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact # Interactive Dashboard
warnings.simplefilter(action='ignore', category=FutureWarning)


# Load dataset
df = pd.read_csv('linear_reg_proj/housing.csv')
# Inpect dataset
df.head()


# Investigate dataset
print(f"Shape of dataset: {df.shape}")
df.info()

# Checking for missing values in total_bedrooms

df['ocean_proximity'].value_counts()

# Encoding the observations of ocean_proximity
df['ocean_proximity'] = (df['ocean_proximity']
                         .map({'<1H OCEAN':1,
                               'INLAND':2,
                               'NEAR OCEAN':3,
                               'NEAR BAY':4,
                               'ISLAND':5})
                         )

# Inspecting the column
df['ocean_proximity'].value_counts()

# Multivariate exploratory analysis
# Using heatmap to detect multicollinearity in the dataset. Drop median_house_value, it's the target vector
corr = df.drop(columns='median_house_value', axis=1).corr()
plt.figure(figsize=[8,5])
mask = np.triu(np.ones_like(corr, dtype= bool))
sns.heatmap(data=corr, annot=True, fmt=".2f", cmap='Blues', mask=mask)
plt.title('Heatmap: Multivariate')
plt.show();


# Plot of multivariate
sns.scatterplot(data=df, x='total_rooms', y='median_house_value', hue='ocean_proximity')
plt.title('Multivariate Plot')
plt.xlabel('Total rooms')
plt.ylabel('House value($)')
plt.show();


# Detecting outliers by boxplot

def box_plot(data: pd.DataFrame):
    """
    This function is intended to plot boxplot for the features of the dataframe
    """

    cols = data.columns

    for c in cols:
        plt.figure(figsize=[10,3])
        plt.subplot(221)
        sns.boxplot(data[str(c)])
        plt.title('Boxplot of ' + str(c))
        plt.xlabel(str(c))
        plt.ylabel('Counts')
        plt.show();


box_plot(df)


# Create a wrangle function
def wrangle(filepath):
    data = pd.read_csv(filepath)

    # Make copy of dataset
    df = data.copy()

    # Subset data: Remove outliers from total rooms
    low, high = df['total_rooms'].quantile([0.1, 0.9])
    mask_total_rooms = df['total_rooms'].between(low,high)
    df = df[mask_total_rooms]

    # Subset data: Drop columns with multicollinearity
    df.drop(columns=['total_bedrooms','population','households'], axis=1, inplace=True)

    # Subset data: Drop columns to avoid data leakage
    df.drop(columns=['longitude','latitude'], axis=1, inplace=True)

    # Subset data: Drop other irrelevant columns
    df.drop(columns=['housing_median_age','median_income'], axis=1, inplace=True)

    # Encoding the observations of ocean_proximity
    ocean_proximity = {'<1H OCEAN':1,
                                'INLAND':2,
                                'NEAR OCEAN':3,
                                'NEAR BAY':4,
                                'ISLAND':5}
    df['ocean_proximity'] = (df['ocean_proximity']
                            .map(ocean_proximity)
                            )
    
    # Re-ordering the index number
    index = range(0,len(df))
    df.index = index
    
    return df


# Re-assess dataset with wrangle function
df = wrangle('linear_reg_proj/housing.csv')
# Select random sample
df.sample(10)


# Target vector
y = df['median_house_value']

# Feature matrix
X = df.drop(columns='median_house_value')

# Check variables
print(f"Shape of target vector: {y.shape}")
print(f"Shape of feature matrix: {X.shape}")


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


y_train_median = np.median(sorted(y_train))
y_baseline_pred = [y_train_median]*len(y_train)
print(f"Median House Value: {y_train_median}")
print(f"Baseline MAE: {round(mean_absolute_error(y_train, y_baseline_pred),5)}")
print(f"Baseline MAPE: {round(mean_absolute_percentage_error(y_train, y_baseline_pred),5)}")


# Instantiate model
model = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=(0.1, 1, 10), cv=5)
)

model

# Check model
assert isinstance(model, Pipeline)


# Fit model
model.fit(X_train, y_train)

# Check fitted model
check_is_fitted(model[0])


# Predict the median house value with the training set
y_train_pred  = model.predict(X_train)


# Check accuracy of the model
print(f"Train MAE: {round(mean_absolute_error(y_train, y_train_pred),5)}")
print(f"Train MAPE: {round(mean_absolute_percentage_error(y_train, y_train_pred),5)}")


# Predicting test target vector
y_test_pred = model.predict(X_test)


# Evaluating model
print(f"Test MAE: {round(mean_absolute_error(y_test, y_test_pred),5)}")
print(f"Test MAPE: {round(mean_absolute_percentage_error(y_test, y_test_pred),5)}")


# Create prediction function
# def make_prediction(total_rooms, ocean_proximity):
    
def make_prediction(total_rooms, ocean_proximity):
    data = {
        'total_rooms': total_rooms,
        'ocean_proximity': ocean_proximity
    }
    df = pd.DataFrame(data=data, index=[0])
    prediction = model.predict(df).round(2)[0]
    return f"Predicted house value: ${prediction}"


# Alternative function
# Create prediction function
# def make_prediction(total_rooms, ocean_proximity):
    
def make_pred():
    data = {
        'total_rooms': int(input("Enter No of rooms:")),
        'ocean_proximity': int(input("Enter location between 1 and 5:"))
    }
    df = pd.DataFrame(data=data, index=[0])
    prediction = model.predict(df).round(2)[0]
    return f"Predicted house value: ${prediction}"


# Make prediction
make_prediction(2215.0,5)

make_pred()


# SLider
interact(
    make_prediction,
    total_rooms=IntSlider(
        min=X_train["total_rooms"].min(),
        max=X_train["total_rooms"].max(),
        value=X_train["total_rooms"].mean(),
    ),
    ocean_proximity=FloatSlider(
        min=X_train["ocean_proximity"].min(),
        max=X_train["ocean_proximity"].max(),
        step=0.01,
        value=X_train["ocean_proximity"].mean(),
    ));


# Save the model
with open('linear_reg_proj/lr_model', 'wb') as f:
    pickle.dump(model, f)

