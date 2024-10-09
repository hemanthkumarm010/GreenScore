import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the dataset from a given filepath."""
    return pd.read_csv(filepath)

def preprocess_data(df):
    """Preprocess the dataset."""
    # Handle missing values
    df.ffill(inplace=True)  # Forward fill for missing values

    # Convert categorical variables into numerical ones
    categorical_columns = ['Location', 'DietType', 'TransportationMode', 'EnergySource', 'HomeType', 'Gender',
                           'LocalFoodFrequency', 'ClothingFrequency', 'SustainableBrands', 
                           'EnvironmentalAwareness', 'CommunityInvolvement', 'UsingPlasticProducts', 
                           'DisposalMethods', 'PhysicalActivities']
    
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Normalize/scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'HomeSize', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    return df

if __name__ == '__main__':
    # Example usage
    df = load_data(r'C:\Users\DSATM\Desktop\sustainable-living-project\data\raw\dataset.csv')
    processed_df = preprocess_data(df)
    processed_df.to_csv(r'C:\Users\DSATM\Desktop\sustainable-living-project\data\processed\processed_data.csv', index=False)
