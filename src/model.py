from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# Load your dataset
data = pd.read_csv(r'C:\Users\DSATM\Desktop\sustainable-living-project\data\processed\processed_data.csv')  # Replace with your actual processed CSV path

# Define your feature columns and target variable
X = data.drop('Rating', axis=1)  # Assuming 'Rating' is your target variable
y = data['Rating']

# Numerical features that need to be scaled
numerical_features = ['Age', 'HomeSize', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption']

# Scale the numerical features using StandardScaler
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Save the scaler for later use in predictions
joblib.dump(scaler, r'C:\Users\DSATM\Desktop\sustainable-living-project\models\scaler.joblib')

# Train your model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, r'C:\Users\DSATM\Desktop\sustainable-living-project\models\random_forest_model.joblib')

print("Model and scaler saved successfully.")
