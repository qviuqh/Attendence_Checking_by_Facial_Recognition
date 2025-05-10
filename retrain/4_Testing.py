import logging
import pandas as pd
import wandb
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import os

# =========================================================================================================
# 4. Testing  
# =========================================================================================================
# name of the artifact related to test dataset
artifact_test_name = "attendance_face_recognition/test.csv:latest"
# name of the model artifact
artifact_model_name = "attendance_face_recognition/model_export:latest"
# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()
# initiate the wandb project
run = wandb.init(project="attendance_face_recognition",job_type="test") 

logger.info("Downloading and reading test artifact")
artifact = run.use_artifact(artifact_test_name)
test_data_path = artifact.download()  # Lưu toàn bộ files về local
df_test = pd.read_csv(os.path.join(test_data_path, "test.csv"))

# Extract the target from the features
logger.info("Extracting target from dataframe")
x_test = df_test.copy()
y_test = x_test.pop("512")

# Download inference artifact
logger.info("Downloading and load the exported model")
model_export_path = run.use_artifact(artifact_model_name).download()
model = joblib.load(os.path.join(model_export_path, "model_export")) 

# predict
logger.info("Infering")
predict = model.predict(x_test)

# Evaluation Metrics
logger.info("Test Evaluation metrics")
acc = accuracy_score(y_test, predict)

logger.info("Test Accuracy: {}".format(acc))

run.summary["Acc"] = acc

print(classification_report(y_test,predict))

run.finish() 
