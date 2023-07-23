---
classes: wide
header:
  overlay_image: /images/heart-disease-project-images/heart-disease-banner.png

title: Heart Disease Classifcation
toc: true
toc_label: "Overview"

---

<style type="text/css">
body {
  font-size: 13pt;
}
</style>

# Predicting Heart Disease Using Machine Learning

**Note:** This was done in a Jupyter Notebook.

This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.

We're going to take the following approach:
1. Problem definition
2. Data
3. Evaluation
4. Features
5. Modelling
6. Experimentation

## 1. Problem Definition

In a statement,
> Given clinical parameters about a patient, can we predict whether or not they have heart disease?

## 2. Data

The original data came from the Cleaveland data from the [UCI Machine Learning Repository.](https://archive.ics.uci.edu/dataset/45/heart+disease)

There is also a version of it available on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## 3. Evaluation

> If we can reach 95% accuracy in predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project. 

## 4. Features

### Data Dictionary

The following are the features we'll use to predict our target variable (heart disease or no heart disease).

1. age - age in years

2. sex - (1 = male; 0 = female)

3. cp - chest pain type
    - 0: Typical angina: chest pain related decrease blood supply to the heart
    - 1: Atypical angina: chest pain not related to heart
    - 2: Non-anginal pain: typically esophageal spasms (non-heart related)
    - 3: Asymptomatic: chest pain not showing signs of disease
    
4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) 
    - anything above 130-140 is typically cause for concern

5. chol - serum cholesterol in mg/dl
    - serum = LDL + HDL + .2 * triglycerides
    - above 200 is cause for concern
    
6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
    - '>126' mg/dL signals diabetes
    
7. restecg - resting electrocardiographic results
    - 0: Nothing to note
    - 1: ST-T Wave abnormality
        - can range from mild symptoms to severe problems
        - signals non-normal heartbeat
    - 2: Possible or definite left ventricular hypertrophy
        - Enlarged heart's main pumping chamber  

8. thalach - maximum heart rate achieved

9. exang - exercise-induced angina (1 = yes; 0 = no)

10. oldpeak - ST depression induced by exercise relative to rest 
    - looks at stress of heart during excercise 
    - unhealthy heart will stress more
    
11. slope - the slope of the peak exercise ST segment
    - 0: Upsloping: better heart rate with excercise (uncommon)
    - 1: Flatsloping: minimal change (typical healthy heart)
    - 2: Downslopins: signs of unhealthy heart

12. ca - number of major vessels (0-3) colored by flourosopy
    - colored vessel means the doctor can see the blood passing through
    - the more blood movement the better (no clots)

13. thal - thalium stress result
    - 1,3: normal
    - 6: fixed defect: used to be defect but ok now
    - 7: reversable defect: no proper blood movement when excercising

14. target - have disease or not (1=yes, 0=no) (= the predicted attribute)

**Note:** No personal identifiable information (PPI) can be found in the dataset.

## Preparing the tools

We're going to use pandas, matplotlib, and NumPy for data analysis and manipulation.

We're also going to use the following Scikit-Learn machine learning models and their metric evaluations to see how well each model performs.

~~~python
# Import all the tools we need

# Regular EDA (exploratory data analysis) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We want our plots to appear inside Jupyter Notebook
%matplotlib inline 

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay 
from sklearn.metrics import auc
~~~

## Load data
~~~python
df = pd.read_csv("heart-disease.csv")
~~~

## Data Exploration (exploratory data analysis or EDA)

The goal here is to find out more about the data and become a subject matter expert on the dataset you're working with.

1. What question(s) are you trying to solve?
2. What kind of data do we have and how do we treat different types?
3. What's missing from the data and how do you deal with it?
4. Where are the outliers and why should you care about them?
5. How can you add, change or remove features to get more out of your data?

We'll first see what our data looks like and what features we are working with.

~~~python
df.head()
~~~

`Output:`

![](/images/heart-disease-project-images/data.png)

Let's also find out how many of each classes there are for our target variable (if a person has heart disease or not).

~~~python
df["target"].value_counts().plot(kind="bar",
                                 color=["salmon","lightblue"])

plt.title("Heart Disease Count")
plt.xlabel("1 = Heart Disease, 0 = No Heart Disease")
plt.ylabel("Amount")
plt.xticks(rotation=0);
~~~

`Output:`

![](/images/heart-disease-project-images/class.png)

This tells us that there are `165` samples with heart disease and `138` samples that do not have heart disease.

Let's also take a look and see what data type each feature is.

~~~python
df.info()
~~~

`Output:`

~~~python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 303 entries, 0 to 302
Data columns (total 14 columns):
 #   Column    Non-Null Count  Dtype  
---  ------    --------------  -----  
 0   age       303 non-null    int64  
 1   sex       303 non-null    int64  
 2   cp        303 non-null    int64  
 3   trestbps  303 non-null    int64  
 4   chol      303 non-null    int64  
 5   fbs       303 non-null    int64  
 6   restecg   303 non-null    int64  
 7   thalach   303 non-null    int64  
 8   exang     303 non-null    int64  
 9   oldpeak   303 non-null    float64
 10  slope     303 non-null    int64  
 11  ca        303 non-null    int64  
 12  thal      303 non-null    int64  
 13  target    303 non-null    int64  
dtypes: float64(1), int64(13)
memory usage: 33.3 KB
~~~

It is important to check for any missing values in our dataset or else a machine learning model won't be able to find patterns.

~~~python
# Check for any missing values
df.isna().sum()
~~~

`Output:`

~~~python
age         0
sex         0
cp          0
trestbps    0
chol        0
fbs         0
restecg     0
thalach     0
exang       0
oldpeak     0
slope       0
ca          0
thal        0
target      0
dtype: int64
~~~

Luckily, there are no missing values in our dataset, so the dataset will be good for our machine learning model.

Next, we'll look at a couple graphs that show the relation of some features to our target variable.

### Heart Disease Frequency according to Sex

~~~python
# Create a plot that compares target column with sex column
pd.crosstab(df.target, df.sex).plot(kind="bar",
                                    figsize=(10,6),
                                    color=["salmon","lightblue"])

plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0);
~~~

`Output:`

![](/images/heart-disease-project-images/sex-frequency.png)


### Age vs. Max Heart Rate for Heart Disease

~~~python
# Create another figure
plt.figure(figsize=(10,6))

# Scatter with positive examples
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="salmon");

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="lightblue");

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age");
plt.ylabel("Maximum Heart Rate")
plt.legend(["Heart Disease", "No Heart Disease"]);
~~~

`Output:`

![](/images/heart-disease-project-images/heart-rate-and-age.png)


### Distribution of Age

~~~python
df.age.plot.hist();
~~~

`Output:`

![](/images/heart-disease-project-images/age-distribution.png)


### Heart Disease Frequency per Chest Pain Type

~~~python
# Create a bar graph to compare chest pain level to target column
pd.crosstab(df.cp, df.target).plot(kind="bar",
                                   figsize=(10,6),
                                   color=["salmon","lightblue"])

# Add some communication
plt.title("Heart Disease Frequency Per Chest Pain Type")
plt.xlabel("Chest Pain Type")
plt.ylabel("Amount")
plt.legend(["No Heart Disease", "Heart Disease"])
plt.xticks(rotation=0);
~~~

`Output:`

![](/images/heart-disease-project-images/chest-pain.png)

cp - chest pain type
- 0: Typical angina: chest pain related decrease blood supply to the heart
    
- 1: Atypical angina: chest pain not related to heart
    
    
- 2: Non-anginal pain: typically esophageal spasms (non heart related)
    
    
- 3: Asymptomatic: chest pain not showing signs of disease

### Correlation Matrix
Finally, let's look at the correlation matrix to see how each feature affects the target variable before we start creating our machine-learning model.

~~~python
# Create correlation matrix
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
~~~

`Output:`

![](/images/heart-disease-project-images/correlation-matrix.png)

## 5. Modelling

We are finally ready to start creating our model. Before we do, let's first split our data into training and testing data so we can evaluate our model properly.

~~~python
# Split data into X and y
X = df.drop("target",axis=1)
y = df["target"]

# Split data into train and test sets
np.random.seed(42)

# Split into train and test test
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)
~~~

Now we've got our data split into training and test sets, it's time to build a machine-learning model.

We'll train it (find the patterns) on the training set.

And we'll test it (use the patterns) on the test set.

We're going to try 3 different machine-learning models:
1. `Logistic Regression`
2. `K-Nearest Neighbors Classifier`
3. `Random Forest Classifier`

We'll create a dictionary to store all the models above and create a function to fit and score the models for us.

~~~python
# Put models in a dictionary
models = {"Logistic Regression": LogisticRegression(),
          "KNN": KNeighborsClassifier(),
          "Random Forest": RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    """
    Fit and evaluate given machine learning models.
    models: a dict of different Scikit-Learn machine learning models
    X_train: training data (no labels)
    X_test: testing data (no labels)
    y_train: training labels
    y_test: test labels
    """
    # Set random seed
    np.random.seed(42)
    
    # Make a dictionary to keep model scores
    model_scores = {}
    
    # Loop through models
    for name, model in models.items():
        # Fit the model to the data
        model.fit(X_train,y_train)
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test,y_test)
    return model_scores
~~~

### Model Comparison

Let's see well our models performed.

**Note:**
* 1.0 = highest score possible
* 0.0 = lowest score possible

~~~python
model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)

model_compare = pd.DataFrame(model_scores, index=["accuracy"])
model_compare.T.plot.bar();
~~~

`Output:`

![](/images/heart-disease-project-images/model-comparison.png)

Models Score:
* `Logistic Regression` performed the best and got the highest score of `88.52%`

* `RandomForestClassifier` scored an `83.61%`

* `KNN` scored a `68.85%` which performed poorly the most.

Now we've got a baseline model and we know a model's first predictions aren't always what we should base our next steps off. What should we do?

Let's look at the following:
* Hyperparameter tuning
* Feature importance
* Confusion matrix
* Cross-validation
* Precision
* Recall
* F1 score
* Classification Report
* ROC curve
* Area under the curve (AUC)

### Hyperparameter tuning (by hand)

We can change the paremeters in the `K-Nearest Neighbors Classifier` to see if it will yield us better results.

It's okay to tune the `K-Nearest Neighbors Classifier` by hand since it only has one parameter.

~~~python
# Let's tune KNN

train_scores = []
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1,21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm 
    knn.fit(X_train, y_train)
    
    # Update training score list
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores list
    test_scores.append(knn.score(X_test, y_test))

plt.plot(neighbors,train_scores, label="Train score")
plt.plot(neighbors, test_scores, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend();

print(f"Maximum KNN score on the test data: {max(test_scores)*100:.2f}%")
~~~

`Output: `

`Maximum KNN score on the test data: 75.41%`

![](/images/heart-disease-project-images/knn.png)

After tuning the` K-Nearest Neighbors Classifier`, its highest score was only `75.41%` which is way below our expectation, so we are going to try tuning another model.

## Hyperparameter tuning with RandomizedSearchCV

We're going to tune:
* `LogisticRegression()`
* `RandomForestClassifier()`

... using RandomizedSearchCV which will make it much easier for us to test many different parameters for our models.

~~~python
# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

# Create hyperparameter grid for RandomForestClassifier
rf_grid = {"n_estimators": np.arange(10, 1000, 50),
           "max_depth": [None, 3, 5, 10],
           "min_samples_split": np.arange(2, 20, 2),
           "min_samples_leaf": np.arange(1, 20, 2)}
~~~

We've got hyperparameter grids setup for each of our models, let's tune them using RandomizedSearchCV.

~~~python
# Tune LogisticRegression
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)
~~~

Now that the model is finished trying out different parameters, let's see which numbers yield the best parameters.

~~~python
# Check for the best parameters
rs_log_reg.best_params_
~~~

`Ouput:`

`{'solver': 'liblinear', 'C': 0.23357214690901212}`

Let's see how well our model performs with our new parameters:

~~~python
rs_log_reg.score(X_test, y_test)
~~~

`Accuracy: 88.52%`

After tuning the `LogisticRegression()`, the score remained the same as before. Before we try tuning any more parameters for it, let's try tuning the `RandomForestClassifier()`.

~~~python
# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)

# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)

# Check for the best parameters
rs_rf.best_params_

# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(X_test, y_test)
~~~

`Accuracy: 86.87`

After tuning the `RandomForestClassifier()`, the score has gone up by about `3%`.

## Hyperparameter Tuning with GridSearchCV

Since our `LogisticRegression` model provides the best scores so far, we'll try and improve them again using GridSearchCV.

~~~python
# Different hyperparameters for our LogisticRegression model
log_reg_grid = {"C": np.logspace(-4, 4, 30),
               "solver": ["liblinear"]}

# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train)

# Check the best parameters
gs_log_reg.best_params_

# Evaluate the grid search LogisticRegression model 
gs_log_reg.score(X_test, y_test)
~~~

`Accuracy: 88.52%`

Even after tuning the `LogisticRegression` model with GridSearchCV, our score has remained the same. However, we should evaluate the model in other metrics.

## Evaluating our tuned machine learning classifier, beyond accuracy

* Receiver Operating Characteristic (ROC) curve
* Area Under ROC Curve (AUC) score
* Confusion matrix
* Classification report
* Precision
* Recall
* F1-score

... and it would be great if cross-validation was used where possible.

To make comparisons and evaluate our trained model, first, we need to make predictions.

~~~python
# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)
~~~

`Output: `

~~~python
array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0,
       0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0], dtype=int64)
~~~

We can now plot the ROC curve that shows how well our model can distinguish between two classes. The most ideal ROC curve will hug the top-left corner. We will also calculate the AUC score.

~~~python
# Calculate roc_curve and auc metric
fpr, tpr, thresholds = roc_curve(y_test, y_preds)
roc_auc = auc(fpr, tpr)

# Display roc_curve
display = RocCurveDisplay(fpr=fpr,
                          tpr=tpr,
                          roc_auc=roc_auc,
                          estimator_name="LogisticalRegressor")
display.plot()
plt.show()
~~~

`Output:`

![](/images/heart-disease-project-images/roc.png)

We will now look at a confusion matrix of our model.

~~~python
# Confusion matrix
sns.set(font_scale=1.5)

def plot_conf_mat(y_test, y_preds):
    """
    Plots a visual looking confusion matrix using Seaborn's heatmap()
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel("Model predictions")
    plt.ylabel("True labels")
    
plot_conf_mat(y_test, y_preds)
~~~

`Output:`

![](/images/heart-disease-project-images/confusion_matrix.png)

This confusion matrix tell us the following:

* True positive = model predicts `1` when truth is `1`
* False positive = model predicts `1` when truth is `0`
* True negative = model predicts `0` when truth is `0`
* False negative = model predicts `0` when truth is `1`

From this confusion matrix, our model has successfully identified `25` negative patients and `29` positive patients. However, there are `3` false negatives and `4` false positives.

Now we've got a ROC curve, an AUC metric, and a confusion matrix. Let's get a classification report as well as cross-validated precision, recall, and f1-score.

~~~python
print(classification_report(y_test, y_preds))
~~~

`Output: `
~~~python
              precision    recall  f1-score   support

           0       0.89      0.86      0.88        29
           1       0.88      0.91      0.89        32

    accuracy                           0.89        61
   macro avg       0.89      0.88      0.88        61
weighted avg       0.89      0.89      0.89        61
~~~

This classification report tells us the following:

* **Precision**: Indicates the proportion of positive identifications (model predicted class 1) which were actually correct.

* **Recall**: Indicates the proportion of actual positives which were correctly classified.

* **F1-score**: A combination of precision and recall. 

* **Support**: Number of samples each metric was calculated on.

* **Accuracy**: The accuracy of the model in decimal form.  

* **Macro average**: The average precision, recall, and f1-score between classes (0 and 1) however, it does not take class imbalance into account.

* **Weighted average**: The average precision, recall, and f1-score between classes (0 and 1) which is calculated with respect to how many samples there are in each class.

### Calculate evaluation metrics using cross-validation

We're going to reevaluate our model using the `cross_val_score` to get a better idea of the accuracy, precision, recall, and f1-score to see how well the model can generalise over the whole dataset.

~~~python
# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver="liblinear")

# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="accuracy")

cv_acc = np.mean(cv_acc)

# Cross-validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="precision")

cv_precision = np.mean(cv_precision)

# Cross-validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="recall")

cv_recall = np.mean(cv_recall)

# Cross-validated f1_score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring="f1")

cv_f1 = np.mean(cv_f1)

# Visualize cross-validated metrics
cv_metrics = pd.DataFrame({"Accuracy": cv_acc,
                           "Precision": cv_precision,
                           "Recall": cv_recall,
                           "F1": cv_f1},
                           index=[0])

cv_metrics.T.plot.bar(title="Cross_validated classifcation metrics",
                      legend=False);
~~~

`Output:`

![](/images/heart-disease-project-images/cv_metrics.png)

Scores for each metric:

* **Accuracy**: `84.47%`

* **Precision**: `82.08%`

* **Recall**: `92.12%`

* **F1-score**: `86.73%`

Overall, not bad scores, and this was done over cross-validation so this gives us more insight into how well our model performed on the dataset as a whole.

### Feature Importance

Feature importance is another way of asking, "Which features contributed most to the outcomes of the model and how did they contribute?"

Finding feature importance is different for each machine learning model. One way to find feature importance is to search for "(MODEL NAME) feature importance."

Let's find the feature importance for our LogisticRegression model.

~~~python
# Fit an instance of LogisticRegression
gs_log_reg.best_params_

clf = LogisticRegression(C= 0.20433597178569418,
                         solver="liblinear")

clf.fit(X_train, y_train)

# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))

# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title="Feature Importance",
                      legend=False);
~~~

`Output:` 

![](/images/heart-disease-project-images/feature_importance.png)

This graph shows us how each feature correlates to our target variable (if a person has heart disease or not). It seems like `cp` and `slope` had the most correlations for heart disease. 

## 6. Conclusion

While we weren't able to reach our evaluation metric goal of `95%` accuracy, we managed to get `88%` which is a great start. Again we can look at the classification report to see how the model performed in different metrics:

~~~python
precision    recall  f1-score   support

           0       0.89      0.86      0.88        29
           1       0.88      0.91      0.89        32

    accuracy                           0.89        61
   macro avg       0.89      0.88      0.88        61
weighted avg       0.89      0.89      0.89        61
~~~

Maybe in the future, I will see if I can try another model or more hyperparameter tuning, but `88%` is still great.

## The Full Code

You can check out all the code together on my [Heart Disease Classifcation repository.](https://github.com/samikamal21/Heart-Disease-Classification)