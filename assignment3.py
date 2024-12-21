
import pandas as pd


file_path = 'file path for dataset'
data = pd.read_csv(file_path)


data.head()

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

X = data.drop(columns=['Price'])
y = data['Price']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kf = KFold(n_splits=5, shuffle=True, random_state=1)


betas = []
r2_scores = []
predicted_values = []


for train_index, test_index in kf.split(X_scaled):
    
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

   
    model = LinearRegression()
    model.fit(X_train, y_train)

   
    y_pred = model.predict(X_test)

    
    r2 = r2_score(y_test, y_pred)

   
    betas.append(model.coef_)
    r2_scores.append(r2)
    predicted_values.append(y_pred)


best_beta_index = np.argmax(r2_scores)
best_beta = betas[best_beta_index]

{
    "best_beta": best_beta,
    "max_r2_score": r2_scores[best_beta_index]
}


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)


final_model = LinearRegression()
final_model.coef_ = best_beta
final_model.fit(X_train, y_train)


y_pred_final = final_model.predict(X_test)


final_r2_score = r2_score(y_test, y_pred_final)

final_r2_score
import numpy as np


X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.44, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.682, random_state=1) # 0.682 * 0.44 ≈ 30%


def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape
    beta = np.zeros(n)
    for _ in range(iterations):
        y_pred = X.dot(beta)
        gradients = -2 / m * X.T.dot(y - y_pred)
        beta -= learning_rate * gradients
    return beta


learning_rates = [0.001, 0.01, 0.1, 1]
iterations = 1000


beta_values = []
val_r2_scores = []
test_r2_scores = []


for lr in learning_rates:
   
    beta = gradient_descent(X_train, y_train.values, learning_rate=lr, iterations=iterations)
    beta_values.append(beta)

   
    y_val_pred = X_val.dot(beta)
    y_test_pred = X_test.dot(beta)

   
    val_r2 = r2_score(y_val, y_val_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    val_r2_scores.append(val_r2)
    test_r2_scores.append(test_r2)


best_index = np.argmax(val_r2_scores)
best_beta = beta_values[best_index]

{
    "best_beta": best_beta,
    "best_learning_rate": learning_rates[best_index],
    "max_val_r2_score": val_r2_scores[best_index],
    "corresponding_test_r2_score": test_r2_scores[best_index]
}
import pandas as pd
import numpy as np


column_names = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
                "num_doors", "body_style", "drive_wheels", "engine_location", "wheel_base",
                "length", "width", "height", "curb_weight", "engine_type", "num_cylinders",
                "engine_size", "fuel_system", "bore", "stroke", "compression_ratio",
                "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]


url = " url for imports-85.data"
data = pd.read_csv(url, names=column_names, na_values="?")

data.fillna(data.median(numeric_only=True), inplace=True)


data.dropna(subset=['price'], inplace=True)
data['price'] = pd.to_numeric(data['price'])
from sklearn.preprocessing import LabelEncoder


data['num_doors'] = data['num_doors'].map({'two': 2, 'four': 4}).astype(float)
data['num_cylinders'] = data['num_cylinders'].map({
    'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'eight': 8, 'twelve': 12
}).astype(float)


data = pd.get_dummies(data, columns=['body_style', 'drive_wheels'])


label_cols = ["make", "aspiration", "engine_location", "fuel_type"]
for col in label_cols:
    data[col] = LabelEncoder().fit_transform(data[col])


data['fuel_system'] = np.where(data['fuel_system'].str.contains('pfi'), 1, 0)


data['engine_type'] = np.where(data['engine_type'].str.contains('ohc'), 1, 0)
from sklearn.preprocessing import StandardScaler


X = data.drop(columns=['price'])
y = data['price']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1)


regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)


initial_r2_score = r2_score(y_test, y_pred)
print(f"Initial R^2 score: {initial_r2_score}")
from sklearn.decomposition import PCA


X_pca = pca.fit_transform(X_scaled)


X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=1)

a
regressor_pca = LinearRegression()
regressor_pca.fit(X_train_pca, y_train)
y_pred_pca = regressor_pca.predict(X_test_pca)


pca_r2_score = r2_score(y_test, y_pred_pca)
print(f"PCA R^2 score: {pca_r2_score}")


if pca_r2_score > initial_r2_score:
    print("PCA improved the model performance.")
else:
    print("PCA did not improve the model performance.")
