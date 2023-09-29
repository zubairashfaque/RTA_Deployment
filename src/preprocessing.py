import pandas as pd
from sklearn import model_selection
import os
import sys
import joblib
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder  # Import LabelEncoder
from sklearn.impute import KNNImputer
import logging

# Configure logging
logging.basicConfig(filename='Project_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(input_file, output_file, ordinal_encoder_mapping_file, target_encoder_mapping_file):  # Add target_encoder_mapping_file as an argument
    try:
        logging.info("-----------------------------------------Preprocessing Started-----------------------------------------")
        # Step 1: Read the CSV file into a DataFrame
        df = pd.read_csv(input_file)
        logging.info("Step 1: CSV file read into DataFrame.")
        print("Step 1: CSV file read into DataFrame.")

        # Step 4: Convert the 'Time' column to datetime and extract hour and minute
        df['Time'] = df['Time'].astype('datetime64[ns]')
        df['hour'] = df['Time'].dt.hour
        df['minute'] = df['Time'].dt.minute
        logging.info("Step 2: 'Time' column converted to datetime.")
        print("Step 2: 'Time' column converted to datetime.")

        # Define a function to categorize the 'hour' into sessions
        def divide_day(x):
            if (x > 4) and (x <= 8):
                return 'Early Morning'
            elif (x > 8) and (x <= 12):
                return 'Morning'
            elif (x > 12) and (x <= 16):
                return 'Noon'
            elif (x > 16) and (x <= 20):
                return 'Evening'
            elif (x > 20) and (x <= 24):
                return 'Night'
            elif x <= 4:
                return 'Late Night'

        # Step 5: Apply the 'divide_day' function to create a 'session' column
        df['session'] = df['hour'].apply(divide_day)
        logging.info("Step 3: 'session' column created based on 'hour.")
        print("Step 3: 'session' column created based on 'hour.")

        # Step 6: Drop the original 'Time' column
        df = df.drop("Time", axis=1)

        # Define the target column
        target = 'Accident_severity'
        selected_features = [
            'Number_of_casualties',
            'minute',
            'Age_band_of_driver',
            'Number_of_vehicles_involved',
            'Light_conditions',
            'Day_of_week',
            'Types_of_Junction',
            'session',
            'hour',
            'Lanes_or_Medians',
            'Accident_severity'
        ]
        df = df[selected_features]

        # Encode the target variable 'Accident_severity'
        target_encoder = LabelEncoder()
        df[target] = target_encoder.fit_transform(df[target])
        joblib.dump(target_encoder, target_encoder_mapping_file)  # Save target encoder mapping
        logging.info("Step 4: Target variable 'Accident_severity' encoded.")
        print("Step 4: Target variable 'Accident_severity' encoded.")

        # Step 8: Perform ordinal encoding on categorical columns (excluding target)
        encoder = OrdinalEncoder()
        categorical_cols = ['Age_band_of_driver', 'Light_conditions', 'Day_of_week', 'Types_of_Junction',
                             'Lanes_or_Medians', 'session']
        df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
        joblib.dump(encoder, 'data/processed/ordinal_encoder.pkl')
        logging.info("Step 5: Categorical columns encoded using ordinal encoding.")
        print("Step 5: Categorical columns encoded using ordinal encoding.")

        # Step 3: Split the data into folds using StratifiedKFold
        kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df[target].values)):
            df.loc[val_idx, "kfold"] = fold
        logging.info("Step 6: Split the data into folds using StratifiedKFold")
        print("Step 6: Split the data into folds using StratifiedKFold")

        # Step 13: Define the number of neighbors for KNN imputation
        k_neighbors = 5

        # Initialize the KNN imputer
        imputer = KNNImputer(n_neighbors=k_neighbors)

        # Step 9: Perform KNN imputation on the dataset
        imputed_data = imputer.fit_transform(df)
        logging.info("Step 7: Missing values imputed using KNNImputer.")
        print("Step 7: Missing values imputed using KNNImputer.")

        # Convert the imputed data back to a DataFrame
        df = pd.DataFrame(imputed_data, columns=df.columns)

        # Step 10: Save the processed DataFrame to the output CSV file
        df.to_csv(output_file, index=False)
        logging.info(f"Step 8: Processed data saved to {output_file}.")
        print(f"Step 8: Processed data saved to {output_file}.")

        logging.info("-----------------------------------------Preprocessing Ended-----------------------------------------")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage:
if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython cross_validation.py data-dir-path output-dir-path\n")
        sys.exit(1)

    train_input_cv = os.path.join(sys.argv[1], "RTA_Dataset.csv")
    train_output_cv = os.path.join(sys.argv[2], "FE_output.csv")

    ordinal_encoder_mapping_file = os.path.join(sys.argv[2], "ordinal_encoder_mapping.json")
    target_encoder_mapping_file = os.path.join(sys.argv[2], "target_encoder.pkl")  # Define target encoder mapping file

    preprocess_data(train_input_cv, train_output_cv, ordinal_encoder_mapping_file, target_encoder_mapping_file)
