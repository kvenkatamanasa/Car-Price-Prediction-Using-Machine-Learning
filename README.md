# Car Price Prediction Using Machine Learning

## 1. Project Overview

The **Car Price Prediction** project is a Machine Learning project that predicts the selling price of a used car based on different features such as kilometers driven, fuel type, seller type, transmission, and vehicle age.

In this project, multiple Machine Learning regression algorithms are trained and compared:

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

After comparing the models using the **R² Score**, the **Gradient Boosting Regressor** is selected as the final model and saved using `joblib`.

---

## 2. Objectives

The main objectives of this project are:

* To analyze used car data.
* To preprocess categorical data.
* To calculate the age of the car.
* To split the dataset into training and testing data.
* To train multiple regression models.
* To compare model performance using R² Score.
* To select the best-performing model.
* To save and load the trained model.
* To predict the selling price of a new car.

---

## 3. Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **XGBoost**
* **Joblib**
* **Jupyter Notebook**

---

## 4. Dataset

The project uses a dataset named:

```text
cardata.csv
```

The dataset contains information about used cars and their selling prices.

### Important Features

| Feature         | Description                   |
| --------------- | ----------------------------- |
| `year`          | Manufacturing year of the car |
| `km_driven`     | Number of kilometers driven   |
| `fuel_type`     | Type of fuel used             |
| `seller_type`   | Type of seller                |
| `transmission`  | Transmission type             |
| `selling_price` | Selling price of the car      |
| `Age`           | Calculated age of the car     |

---

## 5. Data Preprocessing

The dataset is loaded using Pandas:

```python
import pandas as pd

data = pd.read_csv("cardata.csv")
```

The dataset information is checked using:

```python
data.info()
```

### Calculate Car Age

The current year is obtained using Python's `datetime` module.

```python
import datetime

dt = datetime.datetime.now()
data['Age'] = dt.year - data['year']
```

The age of the car is calculated as:

```text
Age = Current Year - Manufacturing Year
```

---

## 6. Encoding Categorical Variables

Machine Learning models require numerical values, so categorical columns are converted into numerical values.

### Fuel Type

```python
data['fuel_type'] = data['fuel_type'].map({
    'Diesel': 0,
    'Petrol': 1,
    'CNG': 2
})
```

Mapping:

| Fuel Type | Value |
| --------- | ----: |
| Diesel    |     0 |
| Petrol    |     1 |
| CNG       |     2 |

### Seller Type

```python
data['seller_type'] = data['seller_type'].map({
    'Individual': 0,
    'Dealer': 1,
    'Trustmark Dealer': 2
})
```

Mapping:

| Seller Type      | Value |
| ---------------- | ----: |
| Individual       |     0 |
| Dealer           |     1 |
| Trustmark Dealer |     2 |

### Transmission

```python
data['transmission'] = data['transmission'].map({
    'Manual': 0,
    'Automatic': 1
})
```

Mapping:

| Transmission | Value |
| ------------ | ----: |
| Manual       |     0 |
| Automatic    |     1 |

---

## 7. Feature and Target Selection

The `year` column is removed because the car's age is already calculated from it.

The `selling_price` column is used as the target variable.

```python
x = data.drop(['year', 'selling_price'], axis=1)
y = data['selling_price']
```

### Input Features

The model uses:

```text
km_driven
fuel_type
seller_type
transmission
Age
```

### Target

```text
selling_price
```

---

## 8. Train-Test Split

The dataset is divided into training and testing data.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Ratio

* **80%** → Training data
* **20%** → Testing data

The `random_state=42` ensures reproducible results.

---

## 9. Machine Learning Models

Four regression algorithms are trained and evaluated.

### 9.1 Linear Regression

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
```

Linear Regression is a simple regression algorithm that attempts to model the relationship between input features and selling price.

---

### 9.2 Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(x_train, y_train)
```

Random Forest combines multiple decision trees to produce a more robust prediction.

---

### 9.3 Gradient Boosting Regressor

```python
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor()
gb.fit(x_train, y_train)
```

Gradient Boosting builds models sequentially, where each new model attempts to improve the errors made by previous models.

---

### 9.4 XGBoost Regressor

```python
from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(x_train, y_train)
```

XGBoost is an optimized gradient boosting algorithm commonly used for regression and classification tasks.

---

## 10. Model Prediction

Predictions are generated using the test dataset:

```python
y_pred1 = lr.predict(x_test)
y_pred2 = rf.predict(x_test)
y_pred3 = gb.predict(x_test)
y_pred4 = xgb.predict(x_test)
```

---

## 11. Model Evaluation

The models are evaluated using the **R² Score**.

```python
from sklearn import metrics

score1 = metrics.r2_score(y_test, y_pred1)
score2 = metrics.r2_score(y_test, y_pred2)
score3 = metrics.r2_score(y_test, y_pred3)
score4 = metrics.r2_score(y_test, y_pred4)
```

The scores are displayed using:

```python
score1, score2, score3, score4
```

### R² Score

The R² score measures how well the model explains the variation in the target variable.

A higher R² score generally indicates better predictive performance.

---

## 12. Final Model

After comparing the models, the **Gradient Boosting Regressor** is used as the final model.

```python
gb_final = GradientBoostingRegressor()
gb_final.fit(x, y)
```

The final model is trained using the complete dataset.

---

## 13. Saving the Model

The trained model is saved using `joblib`.

```python
import joblib

model = joblib.dump(
    gb_final,
    'models/carpriceprediction'
)
```

The saved model can later be loaded without training the model again.

```python
model = joblib.load('models/carpriceprediction')
```

---

## 14. Predicting the Price of a New Car

A new car's information is provided as a Pandas DataFrame.

```python
data_new = pd.DataFrame({
    'km_driven': [22814],
    'fuel_type': [1],
    'seller_type': [1],
    'transmission': [0],
    'Age': [10]
})
```

The model predicts the selling price:

```python
prediction = model.predict(data_new)
```

The predicted price is displayed using:

```python
print("Predicted Selling Price:", prediction[0])
```

---

## 15. Example Input

The example input represents a car with:

| Feature           |    Value |
| ----------------- | -------: |
| Kilometers Driven |   22,814 |
| Fuel Type         |   Petrol |
| Seller Type       |   Dealer |
| Transmission      |   Manual |
| Car Age           | 10 years |

The encoded values used by the model are:

```text
km_driven = 22814
fuel_type = 1
seller_type = 1
transmission = 0
Age = 10
```

---

## 16. Project Workflow

```text
        cardata.csv
             |
             v
       Load Dataset
             |
             v
      Data Exploration
             |
             v
      Calculate Car Age
             |
             v
   Encode Categorical Data
             |
             v
     Feature Selection
             |
             v
       Train-Test Split
             |
             v
   +---------+---------+---------+
   |         |         |         |
   v         v         v         v
Linear    Random    Gradient   XGBoost
Regression Forest   Boosting
   |         |         |         |
   +---------+---------+---------+
             |
             v
       Compare R² Scores
             |
             v
   Select Final Model
             |
             v
   Gradient Boosting
             |
             v
       Save Model
             |
             v
    Predict New Car Price
```

---

## 17. Project Structure

```text
Car-Price-Prediction/
│
├── cardata.csv
├── car_price_prediction.ipynb
│
├── models/
│   └── carpriceprediction
│
└── README.md
```

> The notebook filename can be changed according to the actual filename used in the project.

---

## 18. Installation

Install the required Python libraries:

```bash
pip install pandas scikit-learn xgboost joblib
```

If using Jupyter Notebook:

```bash
pip install notebook
```

---

## 19. How to Run the Project

### Step 1: Clone or download the project

Place all project files in the same project directory.

### Step 2: Make sure the dataset is available

Ensure that:

```text
cardata.csv
```

is present in the project directory.

### Step 3: Install dependencies

```bash
pip install pandas scikit-learn xgboost joblib
```

### Step 4: Open the Jupyter Notebook

```bash
jupyter notebook
```

### Step 5: Run the notebook

Run the cells sequentially:

1. Import libraries
2. Load dataset
3. Explore data
4. Calculate car age
5. Encode categorical columns
6. Prepare features and target
7. Split the dataset
8. Train the models
9. Evaluate the models
10. Select the final model
11. Save the model
12. Load the model
13. Predict a new car price

---

## 20. Advantages

* Uses multiple Machine Learning regression algorithms.
* Compares model performance using R² Score.
* Handles categorical variables through encoding.
* Calculates vehicle age automatically.
* Saves the trained model for future predictions.
* Can be extended into a web application.

---

## 21. Future Enhancements

The project can be further improved by:

* Creating a user interface for price prediction.
* Building a Django or Flask web application.
* Adding more car features such as mileage, engine size, and seats.
* Performing data visualization and exploratory data analysis.
* Performing hyperparameter tuning.
* Comparing additional regression algorithms.
* Adding error metrics such as MAE and RMSE.
* Deploying the model online.

---

## 22. Conclusion

The **Car Price Prediction** project demonstrates how Machine Learning can be used to estimate the selling price of used cars.

The project performs data preprocessing, feature engineering, categorical encoding, model training, model evaluation, and prediction. Four regression algorithms are compared using the R² Score, and the Gradient Boosting Regressor is selected as the final model.

The trained model is saved using `joblib`, allowing it to be loaded later and used to predict the selling price of new cars.

---

## 23. Author

**Kammineni Venkata Manasa**

### Project: Car Price Prediction Using Machine Learning
