import argparse
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from torp_dataset import TORPDataset
import pickle
import io

def check_conditions(s):
    """
    Checks if any part of the string meets the specified criteria.
    
    Args:
        s (str): The string from the 'closestTorPointsInTime' column.
        
    Returns:
        bool: True if any point meets the criteria, False otherwise.
    """
    # The primary check: if the value is the sentinel, no need to process
    if s == '-99900':
        return False

    # Split the string into individual point entries
    points = s.split(';;')
    
    # Iterate through the points to check the conditions.
    # We return True as soon as a single match is found, which is very efficient.
    for p in points:
        try:
            # Split each point into the two values
            dist_str, time_str = p.split(':')
            
            # Convert to numeric types for comparison
            time = float(time_str)
            dist = int(dist_str)
            
            # Check if both conditions are met
            if time <= 100.0 and count <= 60:
                return True
        except (ValueError, IndexError):
            # This handles cases of malformed strings gracefully
            continue
            
    # If we loop through all points and find no match, return False
    return False

#------------------------------------------------------------------------------------------

def mark_and_drop_rows(df):
    """
    Marks rows for dropping based on the 'closestTorPointsInTime' column
    and then drops them.

    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with the specified rows dropped.
    """
    print("Marking rows for potential drop...")
    
    # Apply the check_conditions function to the column.
    # This is the fastest way to perform this operation row-wise.
    df['mark_for_drop'] = df['closestTorPointsInTime'].apply(check_conditions)
    
    # Show the number of rows to be dropped
    rows_to_drop = df['mark_for_drop'].sum()
    print(f"Found {rows_to_drop} rows to be dropped.")
    
    # Drop the rows where 'mark_for_drop' is True
    cleaned_df = df[~df['mark_for_drop']].drop(columns='mark_for_drop')
    
    print("Rows successfully dropped.")
    
    return cleaned_df

#------------------------------------------------------------------------------------------

def train_and_save_model(ds_orig, target_column, output_dir, n_jobs):
    """
    Trains a RandomForestClassifier using a specified number of CPUs and saves the model.

    Args:
        ds_orig (pandas.DataFrame): The whole training and validation dataset 
        target_column (str): The name of the target variable column.
        output_dir (str): The directory where the trained model will be saved.
        n_jobs (int): The number of CPU cores to use. Use -1 to use all available cores.
    """
   
    # Remove any rows within 100km and 60min of a tornado report 
    ds = mark_and_drop_rows(ds_orig)

    # Check if the target column exists in the data
    if target_column not in ds.columns:
        print(f"Error: The target column '{target_column}' was not found in the data.")
        return

    # hard-code torp predictors
    predictor_names = pickle.load(open('torp_predictors.pkl','rb'))

    # get only storms within 100km and 1hr here?

    # Parse out the data we want
    val = ds[ds.year == 2023]
    y_val = val.drop(columns=[target_column])
    columns_to_drop = [col for col in ds.columns if col not in predictor_names]
    X_val = val.drop(columns=columns_to_drop)

    train = ds[ds.year < 2023]
    y_train = train.drop(columns=[target_column])
    X_train = train.drop(columns=columns_to_drop)


    # Define the Random Forest Classifier with the specified parameters
    rf_classifier = RandomForestClassifier(
        bootstrap=True,
        class_weight='balanced_subsample',
        criterion='entropy',
        max_depth=10,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=3,
        min_samples_split=5,
        min_weight_fraction_leaf=0,
        n_estimators=500,
        oob_score=False,
        random_state=None,
        warm_start=False,
        n_jobs=n_jobs  # Use the specified number of jobs
    )

    # Train the classifier
    print(f"Training the Random Forest Classifier using {n_jobs} cores...")
    rf_classifier.fit(X_train, y_train)
    print("Training complete.")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path to save the model
    model_filename = "random_forest_model.joblib"
    model_path = os.path.join(output_dir, model_filename)

    # Save the trained model using joblib
    joblib.dump(rf_classifier, model_path)
    print(f"Model successfully saved to '{model_path}'")

def main():
    """
    Main function to handle command-line arguments and run the training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train a Random Forest Classifier and save the model.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./models",
        help="The directory to save the trained model. Defaults to './models'."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="The number of CPU cores to use for training. Use -1 to use all available cores. Defaults to -1."
    )

    args = parser.parse_args()

    # Define the input file and target column
    dataset = TORPDataset(dirpath='/raid/jcintineo/torcnn/torp_datasets/',
                          years=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
                          dataset_type='Storm_Reports',
    )
    ds = dataset.load_dataframe()
    
    target = "tornado"

    # Run the training pipeline 
    train_and_save_model(ds, target, args.output_dir, args.n_jobs)

if __name__ == "__main__":
    main()
