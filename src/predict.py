import pandas as pd
import joblib

# Load your model and scaler
model = joblib.load(r'C:\Users\DSATM\Desktop\sustainable-living-project\models\random_forest_model.joblib')
scaler = joblib.load(r'C:\Users\DSATM\Desktop\sustainable-living-project\models\scaler.joblib')  # Load the scaler

def map_user_input(user_input):
    # Mapping user input to boolean flags as in your dataset
    transformed_input = {
        'ParticipantID': user_input.get('ParticipantID', 1),  # Default to 1 if not provided
        'Age': user_input['Age'],
        'HomeSize': user_input['HomeSize'],
        'MonthlyElectricityConsumption': user_input['MonthlyElectricityConsumption'],
        'MonthlyWaterConsumption': user_input['MonthlyWaterConsumption'],
        
        # Location mapping
        'Location_Suburban': user_input['Location'] == 'Suburban',
        'Location_Urban': user_input['Location'] == 'Urban',

        # DietType mapping
        'DietType_Mostly Animal-Based': user_input['DietType'] == 'Mostly Animal-Based',
        'DietType_Mostly Plant-Based': user_input['DietType'] == 'Mostly Plant-Based',

        # TransportationMode mapping
        'TransportationMode_Car': user_input['TransportationMode'] == 'Car',
        'TransportationMode_Public Transit': user_input['TransportationMode'] == 'Public Transit',
        'TransportationMode_Walk': user_input['TransportationMode'] == 'Walk',

        # EnergySource mapping
        'EnergySource_Non-Renewable': user_input['EnergySource'] == 'Non-Renewable',
        'EnergySource_Renewable': user_input['EnergySource'] == 'Renewable',

        # HomeType mapping
        'HomeType_House': user_input['HomeType'] == 'House',
        'HomeType_Other': user_input['HomeType'] == 'Other',

        # Gender mapping
        'Gender_Male': user_input['Gender'] == 'Male',
        'Gender_Non-Binary': user_input['Gender'] == 'Non-Binary',
        'Gender_Prefer not to say': user_input['Gender'] == 'Prefer not to say',

        # LocalFoodFrequency mapping
        'LocalFoodFrequency_Often': user_input['LocalFoodFrequency'] == 'Often',
        'LocalFoodFrequency_Rarely': user_input['LocalFoodFrequency'] == 'Rarely',
        'LocalFoodFrequency_Sometimes': user_input['LocalFoodFrequency'] == 'Sometimes',

        # ClothingFrequency mapping
        'ClothingFrequency_Often': user_input['ClothingFrequency'] == 'Often',
        'ClothingFrequency_Rarely': user_input['ClothingFrequency'] == 'Rarely',
        'ClothingFrequency_Sometimes': user_input['ClothingFrequency'] == 'Sometimes',

        # SustainableBrands mapping
        'SustainableBrands_True': user_input['SustainableBrands'] == 'Yes',

        # EnvironmentalAwareness mapping
        'EnvironmentalAwareness_2': user_input['EnvironmentalAwareness'] == 2,
        'EnvironmentalAwareness_3': user_input['EnvironmentalAwareness'] == 3,
        'EnvironmentalAwareness_4': user_input['EnvironmentalAwareness'] == 4,
        'EnvironmentalAwareness_5': user_input['EnvironmentalAwareness'] == 5,

        # CommunityInvolvement mapping
        'CommunityInvolvement_Low': user_input['CommunityInvolvement'] == 'Low',
        'CommunityInvolvement_Moderate': user_input['CommunityInvolvement'] == 'Moderate',

        # UsingPlasticProducts mapping
        'UsingPlasticProducts_Often': user_input['UsingPlasticProducts'] == 'Often',
        'UsingPlasticProducts_Rarely': user_input['UsingPlasticProducts'] == 'Rarely',
        'UsingPlasticProducts_Sometimes': user_input['UsingPlasticProducts'] == 'Sometimes',

        # DisposalMethods mapping
        'DisposalMethods_Composting': user_input['DisposalMethods'] == 'Composting',
        'DisposalMethods_Landfill': user_input['DisposalMethods'] == 'Landfill',
        'DisposalMethods_Recycling': user_input['DisposalMethods'] == 'Recycling',

        # PhysicalActivities mapping
        'PhysicalActivities_Low': user_input['PhysicalActivities'] == 'Low',
        'PhysicalActivities_Moderate': user_input['PhysicalActivities'] == 'Moderate'
    }
    
    return transformed_input

def preprocess_input(user_input, scaler):
    """ Preprocess user input to match the format and scale used in training. """
    # Map user input
    transformed_input = map_user_input(user_input)
    
    # Convert the transformed input into a DataFrame
    input_df = pd.DataFrame([transformed_input])
    
    # Check numerical features before scaling
    numerical_features = ['Age', 'HomeSize', 'MonthlyElectricityConsumption', 'MonthlyWaterConsumption']
    print("Before scaling:", input_df[numerical_features])
    
    # Normalize numerical features using the same scaler as during training
    input_df[numerical_features] = scaler.transform(input_df[numerical_features])
    
    # Check numerical features after scaling
    print("After scaling:", input_df[numerical_features])
    
    return input_df


def predict_sustainability(model, user_input, scaler):
    """ Use the preprocessed input to make predictions using the trained model. """
    # Preprocess the input
    input_df = preprocess_input(user_input, scaler)

    # Print the transformed input for debugging
    print("Transformed Input DataFrame:\n", input_df)

    # Make the prediction
    prediction = model.predict(input_df)
    
    return prediction[0]

if __name__ == "__main__":
    # Example user input (replace this with actual input collection)
    user_input = {
        'ParticipantID': 10,  # Example participant ID
        'Age': 29,
        'HomeSize': 11000000,  # Standardized value for HomeSize (assuming scaling applied)
        'MonthlyElectricityConsumption': 24000000,
        'MonthlyWaterConsumption': 420000,
        'Location': 'Urban',  # Enter values like 'Urban', 'Suburban'
        'DietType': 'Mostly Animal-Based',  # Enter values like 'Mostly Animal-Based', 'Mostly Plant-Based'
        'TransportationMode': 'Car',  # Enter values like 'Car', 'Public Transit', 'Walk'
        'EnergySource': 'Non-Renewable',  # Enter values like 'Non-Renewable', 'Renewable'
        'HomeType': 'House',  # Enter values like 'House', 'Other'
        'Gender': 'Non-Binary',  # Enter values like 'Male', 'Non-Binary', 'Prefer not to say'
        'LocalFoodFrequency': 'Often',  # Enter values like 'Often', 'Rarely', 'Sometimes'
        'ClothingFrequency': 'Often',  # Enter values like 'Often', 'Rarely', 'Sometimes'
        'SustainableBrands': 'Yes',  # Enter 'Yes' or 'No'
        'EnvironmentalAwareness': 2,  # Enter values 2, 3, 4, or 5
        'CommunityInvolvement': 'None',  # Enter 'Low' or 'Moderate'
        'UsingPlasticProducts': 'Often',  # Enter 'Often', 'Rarely', 'Sometimes'
        'DisposalMethods': 'Landfill',  # Enter 'Composting', 'Landfill', 'Recycling'
        'PhysicalActivities': 'None'  # Enter 'Low' or 'Moderate'
    }
    
    # Pass the loaded scaler as the third argument
    prediction = predict_sustainability(model, user_input, scaler)
    
    # Display the sustainability rating
    print(f"Sustainability Rating: {prediction}")
