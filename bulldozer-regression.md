---
classes: wide
header:
  overlay_image: /images/bulldozer-regression-images/bulldozer-banner.jpg

title: Bulldozer Price Regression
toc: true
toc_label: "Overview"

---

<style type="text/css">
body {
  font-size: 13pt;
}
</style>

# Predicting the Sale Price of Bulldozers using Machine Learning

**Note:** This was done in a Jupyter Notebook.

In this notebook, we're going to predict the sale price of bulldozers.

## 1. Problem Definition

> How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?

## 2. Data

The data is downloaded from the [Kaggle Bluebook for Bulldozers competition.](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data)

There are 3 main datasets:

* Train.csv is the training set, which contains data through the end of 2011.

* Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.

* Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

## 3. Evaluation

The evaluation metric is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

**Note:** The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project will be to build a machine-learning model which minimizes RMSLE.

## 4. Features

For this dataset, Kaggle provides a data dictionary that contains information about what each attribute of the dataset means. You can download this file directly from the [Kaggle competition page (account required).](https://www.kaggle.com/account/login?returnUrl=%2Fcompetitions%2Fbluebook-for-bulldozers)

First, we'll import the dataset and start exploring. Since we know the evaluation metric we're trying to minimize, our first goal will be building a baseline model and seeing how it stacks up against the competition.

## Importing the data and preparing it for modelling

~~~python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score
~~~

Now we've got our tools for data analysis ready, we can import the data and start to explore it.

~~~python
# Import training and validation sets
df = pd.read_csv("data/bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False) 

df.info()
~~~

`Output:`

~~~python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 412698 entries, 0 to 412697
Data columns (total 53 columns):
 #   Column                    Non-Null Count   Dtype  
---  ------                    --------------   -----  
 0   SalesID                   412698 non-null  int64  
 1   SalePrice                 412698 non-null  float64
 2   MachineID                 412698 non-null  int64  
 3   ModelID                   412698 non-null  int64  
 4   datasource                412698 non-null  int64  
 5   auctioneerID              392562 non-null  float64
 6   YearMade                  412698 non-null  int64  
 7   MachineHoursCurrentMeter  147504 non-null  float64
 8   UsageBand                 73670 non-null   object 
 9   saledate                  412698 non-null  object 
 10  fiModelDesc               412698 non-null  object 
 11  fiBaseModel               412698 non-null  object 
 12  fiSecondaryDesc           271971 non-null  object 
 13  fiModelSeries             58667 non-null   object 
 14  fiModelDescriptor         74816 non-null   object 
 15  ProductSize               196093 non-null  object 
 16  fiProductClassDesc        412698 non-null  object 
 17  state                     412698 non-null  object 
 18  ProductGroup              412698 non-null  object 
 19  ProductGroupDesc          412698 non-null  object 
 20  Drive_System              107087 non-null  object 
 21  Enclosure                 412364 non-null  object 
 22  Forks                     197715 non-null  object 
 23  Pad_Type                  81096 non-null   object 
 24  Ride_Control              152728 non-null  object 
 25  Stick                     81096 non-null   object 
 26  Transmission              188007 non-null  object 
 27  Turbocharged              81096 non-null   object 
 28  Blade_Extension           25983 non-null   object 
 29  Blade_Width               25983 non-null   object 
 30  Enclosure_Type            25983 non-null   object 
 31  Engine_Horsepower         25983 non-null   object 
 32  Hydraulics                330133 non-null  object 
 33  Pushblock                 25983 non-null   object 
 34  Ripper                    106945 non-null  object 
 35  Scarifier                 25994 non-null   object 
 36  Tip_Control               25983 non-null   object 
 37  Tire_Size                 97638 non-null   object 
 38  Coupler                   220679 non-null  object 
 39  Coupler_System            44974 non-null   object 
 40  Grouser_Tracks            44875 non-null   object 
 41  Hydraulics_Flow           44875 non-null   object 
 42  Track_Type                102193 non-null  object 
 43  Undercarriage_Pad_Width   102916 non-null  object 
 44  Stick_Length              102261 non-null  object 
 45  Thumb                     102332 non-null  object 
 46  Pattern_Changer           102261 non-null  object 
 47  Grouser_Type              102193 non-null  object 
 48  Backhoe_Mounting          80712 non-null   object 
 49  Blade_Type                81875 non-null   object 
 50  Travel_Controls           81877 non-null   object 
 51  Differential_Type         71564 non-null   object 
 52  Steering_Controls         71522 non-null   object 
dtypes: float64(3), int64(5), object(45)
memory usage: 166.9+ MB
~~~

Already, I can see we're going to need to transform our data in order for a machine-learning model to learn any patterns. Let's also take a look at our target variable (`SalePrice` of a bulldozer).

~~~python
df.SalePrice.plot.hist();
~~~

`Output:`

![](/images/bulldozer-regression-images/saleprice.png)

We can see that most of our sale prices of bulldozers are at `$20,000`.

### Parsing dates

When we work with time series data, we want to enrich the time & date component as much as possible. 

We can do that by telling `pandas` which of our columns has dates in it using the `parse_dates` parameter.

~~~python
# Import data again but this time parse data
df = pd.read_csv("data/bluebook-for-bulldozers/TrainAndValid.csv",
                 low_memory=False,
                 parse_dates=["saledate"])

fig, ax = plt.subplots()
ax.scatter(df["saledate"][:1000], df["SalePrice"][:1000])

ax.set_xlabel("Sale Date");
ax.set_ylabel("Sale Price ($)");
~~~

`Output:`

![](/images/bulldozer-regression-images/saledate.png)

We can now examine the dates properly in our dataset for each bulldozer sold.

### Sort DataFrame by sale date

When working with time series data, it's a good idea to sort it by date.

~~~python
# Sort DataFrame in date order
df.sort_values(by=["saledate"], inplace=True, ascending=True)
df.saledate.head(20)
~~~

`Output: `

~~~python
205615   1989-01-17
274835   1989-01-31
141296   1989-01-31
212552   1989-01-31
62755    1989-01-31
54653    1989-01-31
81383    1989-01-31
204924   1989-01-31
135376   1989-01-31
113390   1989-01-31
113394   1989-01-31
116419   1989-01-31
32138    1989-01-31
127610   1989-01-31
76171    1989-01-31
127000   1989-01-31
128130   1989-01-31
127626   1989-01-31
55455    1989-01-31
55454    1989-01-31
Name: saledate, dtype: datetime64[ns]
~~~

### Make a copy of the original DataFrame

We make a copy of the original data frame so when we manipulate the copy, we've still got our original data.

~~~python
# Make a copy of the original DataFrame to perform edits on
df_tmp = df.copy()
~~~

### Add datetime parameters for `saledate` column

~~~python
df_tmp["saleYear"] = df_tmp.saledate.dt.year
df_tmp["saleMonth"] = df_tmp.saledate.dt.month
df_tmp["saleDay"] = df_tmp.saledate.dt.day
df_tmp["saleDayOfWeek"] = df_tmp.saledate.dt.dayofweek
df_tmp["saleDayOfYear"] = df_tmp.saledate.dt.dayofyear

df_tmp.head().T
~~~

`Output:`

![](/images/bulldozer-regression-images/datetime.png)

We have created `5` new rows to our data set that stores all the time metrics from the sale date into their columns. We can now remove the `saledate` column since we've enriched our data frame with time features.

~~~python
# Remove 'saledate' column
df_tmp.drop("saledate", axis=1, inplace=True)
~~~

## 5. Turning Data into Numbers

In order to use a machine-learning model, we first need to convert all strings to numbers and remove `NA` values from the dataset.

### Convert string to categories

One way we can turn all of our data into number is by converting them into pandas categories.

We can check the different datatypes compatiabale with [`pandas`.](https://pandas.pydata.org/pandas-docs/version/1.4/reference/general_utility_functions.html):

First, we need to convert any string data into categories, so our machine-learning model can understand the data.

~~~python
# This will turn all of the strings values into category values 
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype("category").cat.as_ordered()

df_tmp.info()
~~~

`Output: `

~~~python
<class 'pandas.core.frame.DataFrame'>
Int64Index: 412698 entries, 205615 to 409203
Data columns (total 57 columns):
 #   Column                    Non-Null Count   Dtype   
---  ------                    --------------   -----   
 0   SalesID                   412698 non-null  int64   
 1   SalePrice                 412698 non-null  float64 
 2   MachineID                 412698 non-null  int64   
 3   ModelID                   412698 non-null  int64   
 4   datasource                412698 non-null  int64   
 5   auctioneerID              392562 non-null  float64 
 6   YearMade                  412698 non-null  int64   
 7   MachineHoursCurrentMeter  147504 non-null  float64 
 8   UsageBand                 73670 non-null   category
 9   fiModelDesc               412698 non-null  category
 10  fiBaseModel               412698 non-null  category
 11  fiSecondaryDesc           271971 non-null  category
 12  fiModelSeries             58667 non-null   category
 13  fiModelDescriptor         74816 non-null   category
 14  ProductSize               196093 non-null  category
 15  fiProductClassDesc        412698 non-null  category
 16  state                     412698 non-null  category
 17  ProductGroup              412698 non-null  category
 18  ProductGroupDesc          412698 non-null  category
 19  Drive_System              107087 non-null  category
 20  Enclosure                 412364 non-null  category
 21  Forks                     197715 non-null  category
 22  Pad_Type                  81096 non-null   category
 23  Ride_Control              152728 non-null  category
 24  Stick                     81096 non-null   category
 25  Transmission              188007 non-null  category
 26  Turbocharged              81096 non-null   category
 27  Blade_Extension           25983 non-null   category
 28  Blade_Width               25983 non-null   category
 29  Enclosure_Type            25983 non-null   category
 30  Engine_Horsepower         25983 non-null   category
 31  Hydraulics                330133 non-null  category
 32  Pushblock                 25983 non-null   category
 33  Ripper                    106945 non-null  category
 34  Scarifier                 25994 non-null   category
 35  Tip_Control               25983 non-null   category
 36  Tire_Size                 97638 non-null   category
 37  Coupler                   220679 non-null  category
 38  Coupler_System            44974 non-null   category
 39  Grouser_Tracks            44875 non-null   category
 40  Hydraulics_Flow           44875 non-null   category
 41  Track_Type                102193 non-null  category
 42  Undercarriage_Pad_Width   102916 non-null  category
 43  Stick_Length              102261 non-null  category
 44  Thumb                     102332 non-null  category
 45  Pattern_Changer           102261 non-null  category
 46  Grouser_Type              102193 non-null  category
 47  Backhoe_Mounting          80712 non-null   category
 48  Blade_Type                81875 non-null   category
 49  Travel_Controls           81877 non-null   category
 50  Differential_Type         71564 non-null   category
 51  Steering_Controls         71522 non-null   category
 52  saleYear                  412698 non-null  int64   
 53  saleMonth                 412698 non-null  int64   
 54  saleDay                   412698 non-null  int64   
 55  saleDayOfWeek             412698 non-null  int64   
 56  saleDayOfYear             412698 non-null  int64   
dtypes: category(44), float64(3), int64(10)
memory usage: 63.2 MB
~~~

Now our data set no longer contains strings and to show that our model can access these features, categories have a code attribute meaning we now have a way to access all our data in the form of numbers.

~~~python
df_tmp.state.cat.codes
~~~

`Output:`

~~~python
205615    43
274835     8
141296     8
212552     8
62755      8
          ..
410879     4
412476     4
411927     4
407124     4
409203     4
Length: 412698, dtype: int8
~~~

Thanks to categories, all our data is in the form of numbers however, we still have a bunch of missing data we need to deal with.

### Fill numerical missing values first

We'll first see which numeric columns have missing values and null values.

~~~python
# Check for which numeric columns have missing values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)

# Check for which numeric columns have null values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
~~~

`Output: `

Columns with missing values:

* `SalesID`
* `SalePrice`
* `MachineID`
* `ModelID`
* `datasource`
* `auctioneerID`
* `YearMade`
* `MachineHoursCurrentMeter`
* `saleYear`
* `saleMonth`
* `saleDay`
* `saleDayOfWeek`
* `saleDayOfYear`


Columns with null values:

* `auctioneerID`
* `MachineHoursCurrentMeter`

To deal with the missing values, we'll just fill in any missing row with the median.

~~~python
# Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label+"_is_missing"] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())
~~~

### Filling and turning categorical variables into numbers

We'll now turn out categories into nodes and make new columns that tell us which features were missing.

~~~python
# Turn categorical variables into numbers and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label+"_is_missing"] = pd.isnull(content)
        # Turn categories into numbers and add +1 to make all our numbers positive in our data frame
        df_tmp[label] = pd.Categorical(content).codes + 1
~~~

Now let's see if there are any missing values in our data set.

~~~python
df_tmp.isna().sum()
~~~

`Output: `

~~~python
SalesID                         0
SalePrice                       0
MachineID                       0
ModelID                         0
datasource                      0
                               ..
Backhoe_Mounting_is_missing     0
Blade_Type_is_missing           0
Travel_Controls_is_missing      0
Differential_Type_is_missing    0
Steering_Controls_is_missing    0
Length: 103, dtype: int64
~~~

### Save preprocessed data

A good idea is to save our processed data so if you were to open this Jupyter Notebook, you wouldn't have to run all the code above.

~~~python
# Export current tmp dataframe
df_tmp.to_csv("data/bluebook-for-bulldozers/train_tmp_processed.csv",
              index=False)

# Import preprocessed data
df_tmp = pd.read_csv("data/bluebook-for-bulldozers/train_tmp_processed.csv",
                     low_memory=False)
~~~

Now, we are ready to create a machine-learning model after pre-processing our data.

## 6. Modelling

We've done enough exploratory data analysis (EDA) but let's start to do some model-driven EDA.

### Splitting data into train/validation sets

~~~python
# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]

# Split data into X & y
X_train, y_train = df_train.drop("SalePrice", axis=1), df_train["SalePrice"]
X_valid, y_valid = df_val.drop("SalePrice", axis=1), df_val["SalePrice"]
~~~

Now that we have split our data, we can finally start evaluating our model on the data set.

### Building an evaluation function

Now we'll build a function to test our machine-learning model on different metrics.

~~~python
# Create evaluation function for RMSLE (root-mean-squared-log-error)
def rmsle(y_test, y_preds):
    """
    Calculates root-mean-squared-log-error between predictions and true labels.
    """
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluate model on a few different metrics
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {"Train MAE": mean_absolute_error(y_train, train_preds),
              "Valid MAE": mean_absolute_error(y_valid, val_preds),
              "Training RMSLE": rmsle(y_train, train_preds),
              "Valid RMSLE": rmsle(y_valid, val_preds),
              "Training R^2": r2_score(y_train, train_preds),
              "Valid R^2": r2_score(y_valid, val_preds)}

    return scores
~~~

## Testing our model on a subset (to tune the hyperparameters)

We'll be using the `RandomForestRegressor` model and will train it on a subset of the data so experimentation does not take as long since there are over `400,000` samples.

~~~python
# Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)

%%time
# Cutting down on the max number of samples each estimator can see improves training time
model.fit(X_train, y_train)
~~~

`Output:`

`CPU times: total: 36.8 s`
`Wall time: 7.41 s`
`RandomForestRegressor(max_samples=10000, n_jobs=-1, random_state=42)`

Since we trained it on a subset, it only took `7.41` seconds. 

Now let's see how our model performed on the following metrics below:

* `MAE`: A way to measure how far apart predictions are from the actual values in a set of data.

* `RMSLE`: A way to measure the accuracy of predictions when the values being predicted can vary greatly in magnitude.

* `R^2`: A score that tells you how well your model fits the data.

~~~python
show_scores(model)
~~~

`Output: `

* `Train MAE: 5561.2988092240585`
* `Valid MAE: 7177.26365505919`
* `Training RMSLE: 0.257745378256977`
* `Valid RMSLE: 0.29362638671089003`
* `Training R^2: 0.8606658995199189`
* `Valid R^2: 0.8320374995090507`

So for our `Valid MAE`, our model was about `$7177.26` off from the actual price and for our `RMSLE`, the model's predictions differ from the actual prices by a factor of `exp(0.29) ≈ 1.336`. And finally, the `Valid R^2` score was about `83.23%` accuracy.

Our valid scores are lower than the training data so we did not overfit the data. Since we did only limit the training to `10,000` samples the scores are going to be worse, but overall not bad. However before we try training on the full dataset, let's see if we can tune this model to get better results.

### Hyperparameter tuning with RandomizedSearchCV

We will now tune our model with different parameters.

~~~python
%%time

# Different RandomForestRegressor hyperparameters
rf_grid = {"n_estimators": np.arange(10, 100, 10),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20,2),
           "min_samples_leaf": np.arange(1, 20, 2),
           "max_features": [0.5, 1, "sqrt", "auto"],
           "max_samples": [10000]}

# Instantiate RandomizedSearchCV model
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                    random_state=42),
                             param_distributions=rf_grid,
                             n_iter=5,
                             cv=5,
                             verbose=True,
                             random_state=42)

# Fit the RandomizedSearchCV model
rs_model.fit(X_train, y_train)
~~~

`Output:`

`Fitting 5 folds for each of 5 candidates, totalling 25 fits`
`CPU times: total: 20.7 s`
`Wall time: 38.1 s`

After fitting the `RandomizedSearchCV` model, let's see which parameters were the best.

~~~python
rs_model.best_params_
~~~

`Output:`

~~~python
{'n_estimators': 60,
 'min_samples_split': 12,
 'min_samples_leaf': 1,
 'max_samples': 10000,
 'max_features': 1,
 'max_depth': None}
~~~

Now we'll evalute the model with our current parameters and show the before and after scores.

~~~python
# Evaluate the RandomizedSearch model
show_scores(rs_model)
~~~

Before:

* `Train MAE: 5561.2988092240585`
* `Valid MAE: 7177.26365505919`
* `Training RMSLE: 0.257745378256977`
* `Valid RMSLE: 0.29362638671089003`
* `Training R^2: 0.8606658995199189`
* `Valid R^2: 0.8320374995090507`

After:

* `Train MAE: 8891.655193695899`
* `Valid MAE: 11313.549827603978`
* `Training RMSLE: 0.39237058829152377`
* `Valid RMSLE: 0.4484052255971633`
* `Training R^2: 0.6825547447927643`
* `Valid R^2: 0.6361784117343763`

Our scores have actually gone down after tuning, but that just means we may need to try more iterations of finding better parameters.

### Train a model with the best hyperparameters

**Note:** These were found after `100` iterations of `RandomizedSearchCV`. We'll also show the scores from the previous experiment with our best parameters.

~~~python
%%time

# Most ideal hyperparameters
ideal_model = RandomForestRegressor(n_estimators=40,
                                    min_samples_leaf=1,
                                    min_samples_split=14,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None,
                                    random_state=42)

# Fit the ideal model
ideal_model.fit(X_train, y_train)

# Scores for the ideal model on all the data
show_scores(ideal_model)
~~~

Baseline Scores:

* `Train MAE: 5561.2988092240585`
* `Valid MAE: 7177.26365505919`
* `Training RMSLE: 0.257745378256977`
* `Valid RMSLE: 0.29362638671089003`
* `Training R^2: 0.8606658995199189`
* `Valid R^2: 0.8320374995090507`

Previous Experiment:

* `Train MAE: 8891.655193695899`
* `Valid MAE: 11313.549827603978`
* `Training RMSLE: 0.39237058829152377`
* `Valid RMSLE: 0.4484052255971633`
* `Training R^2: 0.6825547447927643`
* `Valid R^2: 0.6361784117343763`

Best Hyperparameters:

* `Train MAE: 2953.8161137163484`
* `Valid MAE: 5951.247761444453`
* `Training RMSLE: 0.14469006962371858`
* `Valid RMSLE: 0.24524163989538328`
* `Training R^2: 0.9588145522577225`
* `Valid R^2: 0.8818019502450094`

The model has scored much higher than the previous last `2` experiments in all metrics.

## Make predictions on test data

We'll first need to import our test data.

~~~python
# Import test data
df_test = pd.read_csv("data/bluebook-for-bulldozers/Test.csv",
                      low_memory=False,
                      parse_dates=["saledate"])

df_test.head()
~~~

Before we can make predictions on the test data, we need to preprocess it first just like our training and validation datasets.

~~~python
def preprocess_data(df):
    """
    Performs transformations on df and returns transformed df.
    """

    # Add datetime parameters for saledate column
    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    
    # Drop saledate column
    df.drop("saledate", axis=1, inplace=True)
    
    for label, content in df.items():
        # Find the columns which contain strings and turn them into categories
        if pd.api.types.is_string_dtype(content):
            df[label] = content.astype("category").cat.as_ordered()
            
        # Fill numeric rows with the median
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                df[label+"_is_missing"] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
        
        # Turn categorical variables into numbers and fill missing
        if not pd.api.types.is_numeric_dtype(content):
            # Add binary column to indicate whether sample had missing value
            df[label+"_is_missing"] = pd.isnull(content)
            # Turn categories into numbers and add +1 to make all our numbers positive in our data frame
            df[label] = pd.Categorical(content).codes + 1
        
    return df

# Process the test data
df_test = preprocess_data(df_test)
~~~

There is still a problem. The test data is slightly different than the training data so we need to see which columns are different.

~~~python
# We can find how the columns differ using sets
set(X_train.columns) - set(df_test.columns)
~~~

`Output: set(auctioneerID_is_missing)`

This tells us that we need to add the `auctioneerID_is_missing` column to our test data. We'll fix this by adding the `X_train` columns to the test data.

~~~python
# Manually adjust df_test to have auctioneerID_is_missing column
df_test = df_test[X_train.columns]
~~~

Finally, our test data frame has the same features as our training data frame, we can make predictions. Then we'll create a `DataFrame` to show the predictions.

~~~python
# Make predictions on the test data
test_preds = ideal_model.predict(df_test)

# Format predictions
df_preds = pd.DataFrame()
df_preds["SaleID"] = df_test["SalesID"]
df_preds["SalePrice"] = test_preds
df_preds
~~~

`Output: `

![](/images/bulldozer-regression-images/predictions.png)

Finally, let's wrap up this project by looking at the important features.

### Feature Importance

Feature importance seeks to figure out which different attributes of the data were most importance when it comes to predicting the **target variable** `(SalePrice)`.

~~~python
# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({"features": columns,
                        "feature_importances": importances})
         .sort_values("feature_importances", ascending=False)
         .reset_index(drop=True))
    
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df["features"][:n], df["feature_importances"][:n])
    ax.set_ylabel("Features")
    ax.set_xlabel("Feature importance")
    ax.invert_yaxis()

plot_features(X_train.columns, ideal_model.feature_importances_)
~~~

`Output:`

![](/images/bulldozer-regression-images/features.png)

From the graph, `YearMade` and `ProductSize` seem to have the most correlation with the `SalePrice` which is our target variable.

## Conclusion

Overall, our model was able to perform well in all the metrics we evaluated. We'll look over the scores one last time:

* `Train MAE: 2953.8161137163484`
* `Valid MAE: 5951.247761444453`
* `Training RMSLE: 0.14469006962371858`
* `Valid RMSLE: 0.24524163989538328`
* `Training R^2: 0.9588145522577225`
* `Valid R^2: 0.8818019502450094`

In the future, I may try more tuning or use another machine learning model but nevertheless, I am satisfied with these results.

## The Full Code

You can check out all the code together on my [Bulldozer Price Regression repository.](https://github.com/samikamal21/Bulldozer-Price-Regression)