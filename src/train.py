# Import necessary libraries
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from collections import OrderedDict
from sklearn.metrics import f1_score
import logging
import pickle
import bz2
import os

# Configure logging
logging.basicConfig(filename='Project_log.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the file paths
test_file = 'data/processed/sample_data/oversampler_adasyn_0_test.csv'
train_file = 'data/processed/sample_data/oversampler_adasyn_0_train.csv'
selected_features_file = 'data/selected_features_test.csv'
model_file = 'model/extra_trees_model.pkl'

try:
    logging.info(
        "-----------------------------------------Training Started-----------------------------------------")
    # Load the training data
    train_data = pd.read_csv(train_file)

    # Select the specified features
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
        'Lanes_or_Medians'
    ]

    X_train = train_data[selected_features]
    y_train = train_data['Accident_severity']

    # Define the Extra Tree model with hyperparameters
    hyperparameters = OrderedDict([
        ('bootstrap', False),
        ('max_depth', None),
        ('max_features', 'sqrt'),
        ('min_samples_leaf', 1),
        ('min_samples_split', 2),
        ('n_estimators', 120)
    ])

    model = ExtraTreesClassifier(**hyperparameters)

    # Train the model
    model.fit(X_train, y_train)

    # Load the test data
    test_data = pd.read_csv(test_file)

    # Select the same features for the test data
    X_test = test_data.drop(['Accident_severity', 'kfold'], axis=1)
    X_test = test_data[selected_features]
    y_test = test_data['Accident_severity']

    # Predict using the trained model
    y_pred = model.predict(X_test)

    # Calculate F1-weighted score
    f1_weighted_score = f1_score(y_test, y_pred, average='weighted')

    # Print the F1-weighted score
    logging.info("F1 Weighted Score: {}".format(f1_weighted_score))
    print("F1 Weighted Score:", f1_weighted_score)

    # Save the selected features to a CSV file
    X_test.to_csv(selected_features_file, index=False)
    logging.info("Selected features saved to: {}".format(selected_features_file))
    print("Selected features saved to:", selected_features_file)

    # Save the trained model using pickle
    model_pickle_file = 'model/extra_trees_model.pkl'
    with open(model_pickle_file, 'wb') as f_out:
        pickle.dump(model, f_out)

    logging.info("Model saved using pickle to: {}".format(model_pickle_file))
    print("Model saved using pickle to:", model_pickle_file)

    # Save the trained model using bzip2 compression
    compressed_model_file = 'model/extra_trees_model.pkl.bz2'
    with bz2.BZ2File(compressed_model_file, 'wb') as f_out:
        pickle.dump(model, f_out)

    logging.info("Compressed model saved to: {}".format(compressed_model_file))
    print("Compressed model saved to:", compressed_model_file)
    # Remove the pickle file after it's been successfully compressed
    os.remove(model_pickle_file)
    logging.info("Pickle model file removed: {}".format(model_pickle_file))
    print("Pickle model file removed:", model_pickle_file)
    logging.info(
        "-----------------------------------------Training Ended-----------------------------------------")
except Exception as e:
    logging.error("An error occurred: {}".format(str(e)))
