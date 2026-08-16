# California house price prediction using machine learning

## Project overview

This repository presents a complete machine learning regression pipeline for predicting California housing prices using the California Housing Dataset from Scikit-learn. The project compares classical linear models with ensemble learning methods and emphasizes model evaluation, feature engineering, and explainability.

## Problem statement

Predict the median house value (`MedHouseVal`) for California districts using demographic and housing-related features.

This is a **supervised regression problem** because the target variable is continuous.

## Dataset

- **Source:** Scikit-learn (`fetch_california_housing`)
- **Samples:** 20,640
- **Features:** 8 original numerical features
- **Target:** Median house value

### Original features

| Feature | Description |
|--------|-------------|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population |
| AveOccup | Average occupancy |
| Latitude | Latitude |
| Longitude | Longitude |

## Machine learning workflow

Business understanding

↓

Exploratory data analysis

↓

Data cleaning

↓

Feature engineering

↓

Train-test split

↓

Model training

↓

Cross validation

↓

Hyperparameter tuning

↓

Ensemble learning

↓

Feature importance analysis

↓

Final evaluation

## Exploratory data analysis

The dataset was analyzed using:

- descriptive statistics
- correlation analysis
- target distribution
- outlier inspection
- feature relationship analysis

## Feature engineering

Two engineered features were created to improve predictive performance:

### Rooms per bedroom

RoomsPerBedroom = AveRooms / AveBedrms

### Population density

PopulationDensity = Population / AveOccup

These features capture housing structure and neighborhood density more effectively than the original variables alone.

## Models evaluated

### Linear regression

Baseline regression model.

### Ridge regression

L2-regularized linear regression.

### Lasso regression

L1-regularized regression with feature selection capability.

### Random forest regressor

Bagging-based ensemble method capable of modeling nonlinear relationships.

### Extra trees regressor

Highly randomized ensemble of decision trees.

## Model evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Cross-validation

## Results

|Model|MAE|MSE|RMSE|R2|Training Time|
|---|---|---|---|---|---|
|Linear Regression|0\.5077709376611416|0\.48822999046997934|0\.6987345636720567|0\.6274216686721865|0\.028338909149169922|
|Ridge|0\.5077709381896245|0\.4882299936004946|0\.6987345659121886|0\.6274216662832262|0\.007593631744384766|
|Lasso|0\.5078290808741284|0\.48855752950983034|0\.6989689045371262|0\.6271717169049981|0\.10853981971740723|
|Random Forest|0\.32883211637596915|0\.2552540887669524|0\.5052267696460199|0\.8052103633251911|35\.30932903289795|
|Extra Trees|0\.342399325145349|0\.26844021129139634|0\.5181121609182672|0\.7951477624552359|15\.11094069480896|


## Key findings

- Ensemble methods outperformed linear models.
- Random Forest achieved the best predictive performance.
- Feature engineering improved model performance.
- Tree-based models captured nonlinear relationships more effectively.

## Model explainability

Feature importance analysis was performed using:

- Random Forest feature importance

Diagnostic plots included:

- predicted vs actual
- residual plots
- residual distribution

## Repository structure

california-house-price-prediction/

├── notebooks/

├── figures/

├── results/

├── data/

├── LICENSE

├── README.md

└── requirements.txt

## Technologies used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook


## Reproducibility

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
jupyter notebook notebooks/california_house_price_prediction_multimodel_comparison.ipynb
```

## Author

**Mohammad Nazmul Islam**

Research interests: Machine Learning, Deep Learning, Computer Vision, Medical Imaging, Explainable AI
