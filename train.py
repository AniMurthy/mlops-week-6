import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os
import subprocess 

DATA_CSV_PATH = 'iris_1.csv'
ARTIFACTS_DIR = 'artifacts'
MODEL_FILENAME = 'model.joblib'
METRICS_FILENAME = 'metrics.txt'

MODEL_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
METRICS_PATH = os.path.join(ARTIFACTS_DIR, METRICS_FILENAME)

# GCS_URI = 'iitmbs-mlops-ani-arboreal-harbor-461417-u1-week1/my-models/mlops-demo'

# GCP_BUCKET_NAME = 'iitmbs-mlops-ani-arboreal-harbor-461417-u1-week1' # e.g., 'my-iris-model-bucket-123'
# GCP_MODEL_BLOB_NAME = f'my-models/mlops-demo' # Path within the bucket
# GCP_METRICS_BLOB_NAME = f'{ARTIFACTS_DIR}/{METRICS_FILENAME}'

# def upload_to_gcs(local_path, gcs_destination_uri):
#     """Uploads a file to the Google Cloud Storage bucket."""
#     # storage_client = storage.Client()
#     # bucket = storage_client.bucket(bucket_name)
#     # blob = bucket.blob(destination_blob_name)
#     # blob.upload_from_filename(source_file_name)
#     # print(f"File {source_file_name} uploaded to gs://{bucket_name}/{destination_blob_name}")
    
#     command = ["gsutil", "cp", local_path, gcs_destination_uri]
#     print(f"Executing command: {' '.join(command)}")
#     result = subprocess.run(command, check=True, capture_output=True, text=True)
#     print(f"Successfully uploaded {local_path} to {gcs_destination_uri} using gsutil.")
    
def train_model():
    """
    Loads iris data, trains a Decision Tree, and saves the model
    and metrics locally.
    """
    #Load Data
    data = pd.read_csv(DATA_CSV_PATH)
    print(f"Data loaded successfully. Shape: {data.shape}")
   
    #Define Features and Target
    X_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    y_target = 'species'

    #Split Data
    X = data[X_features]
    y = data[y_target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    #Train the Model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model trained: {model}")

    #Evaluate
    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print(f"Model Test Accuracy: {accuracy:.4f}")

    #Save Artifacts
    joblib.dump(model, MODEL_PATH)
    print(f"Trained model saved to: {MODEL_PATH}")
    
    with open(METRICS_PATH, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
    print(f"Metrics saved to: {METRICS_PATH}")
    
    # print("\n--- Uploading artifacts to GCS ---")
    # upload_to_gcs(MODEL_PATH,GCS_URI)

    

if __name__ == "__main__":
    print("--- Running Iris Decision Tree Model Training ---")
    train_model()
    print("--- Model Training and Local Saving Complete ---")
