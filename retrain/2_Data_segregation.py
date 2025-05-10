import logging
import tempfile
import pandas as pd
import os
import wandb
from sklearn.model_selection import train_test_split


# =========================================================================================================
# 2. Data_segregation 
# =========================================================================================================

# ratio used to split train and test data
test_size = 0.20
# seed used to reproduce purposes
seed = 42
# reference (column) to stratify the data
stratify = "512"
# name of the input artifact
artifact_input_name = "attendance_face_recognition/embedding_data.csv:latest"
# type of the artifact
artifact_type = "segregated_data"
# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# initiate wandb project
run = wandb.init(project="attendance_face_recognition", job_type="split_data")

logger.info("Downloading and reading artifact")
artifact = run.use_artifact(artifact_input_name)
artifact_dir = artifact.download()  # Lưu toàn bộ files về local
df = pd.read_csv(os.path.join(artifact_dir, "embedding_data.csv")) 

# Split firstly in train/test, then we further divide the dataset to train and validation
logger.info("Splitting data into train and test")
splits = {}

splits["train"], splits["test"] = train_test_split(df,
                                                   test_size=test_size,
                                                   random_state=seed,
                                                   stratify=df[stratify])

# Save the artifacts. We use a temporary directory so we do not leave any trace behind
with tempfile.TemporaryDirectory() as tmp_dir:

    for split, df in splits.items():

        # Make the artifact name from the name of the split plus the provided root
        artifact_name = f"{split}.csv"

        # Get the path on disk within the temp directory
        temp_path = os.path.join(tmp_dir, artifact_name)

        logger.info(f"Uploading the {split} dataset to {artifact_name}")

        # Save then upload to W&B
        df.to_csv(temp_path,index=False)

        artifact = wandb.Artifact(name=artifact_name,
                                  type=artifact_type,
                                  description=f"{split} split of dataset {artifact_input_name}",
        )
        artifact.add_file(temp_path)

        logger.info("Logging artifact")
        run.log_artifact(artifact)

        # This waits for the artifact to be uploaded to W&B. If you
        # do not add this, the temp directory might be removed before
        # W&B had a chance to upload the datasets, and the upload
        # might fail
        artifact.wait()

# close the run
# waiting a while after run the previous cell before execute this
run.finish() 

