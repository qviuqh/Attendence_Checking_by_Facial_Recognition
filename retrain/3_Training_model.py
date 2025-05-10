import logging
import wandb
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# =========================================================================================================
# 3. Training_model  
# =========================================================================================================
# ratio used to split train and validation data
val_size = 0.2
# seed used to reproduce purposes
seed = 42
# reference (column) to stratify the data
stratify = "512"
# name of the input artifact
artifact_input_name = "attendance_face_recognition/train.csv:latest"
# type of the artifact
artifact_type = "Train"
# configure logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s",
                    datefmt='%d-%m-%Y %H:%M:%S')

# reference for a logging obj
logger = logging.getLogger()

# initiate the wandb project
run = wandb.init(project="attendance_face_recognition",job_type="train")

logger.info("Downloading and reading train artifact")
artifact = run.use_artifact(artifact_input_name)
artifact_dir = artifact.download()  # Lưu toàn bộ files về local

df_train = pd.read_csv(os.path.join(artifact_dir, "train.csv"))

# Spliting train.csv into train and validation dataset
logger.info("Spliting data into train/val")
# split-out train/validation and test dataset
x_train, x_val, y_train, y_val = train_test_split(df_train.drop(labels=stratify,axis=1),
                                                  df_train[stratify],
                                                  test_size=val_size,
                                                  random_state=seed,
                                                  shuffle=True,
                                                  stratify=df_train[stratify])

logger.info("x train: {}".format(x_train.shape))
logger.info("y train: {}".format(y_train.shape))
logger.info("x val: {}".format(x_val.shape))
logger.info("y val: {}".format(y_val.shape))

sweep_config = {
    "method": "random",  # random search
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "n_estimators": {"values": [50, 100, 200, 300]},
        "max_depth": {"values": [5, 10, 20, None]},
        "min_samples_split": {"values": [2, 5, 10]},
        "min_samples_leaf": {"values": [1, 2, 5]},
        "max_features": {"values": ["sqrt", "log2", None]},
        "bootstrap": {"values": [True, False]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="attendance_face_recognition")

def train():
    # Mỗi lần agent chạy, sẽ tạo 1 run mới với các config khác nhau
    with wandb.init() as run:
        config = run.config
        
        # Xây dựng và train model với config hiện tại
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            max_features=config.max_features,
            bootstrap=config.bootstrap,
            random_state=42,
            n_jobs=-1
        )
        model.fit(x_train, y_train)
        
        # Đánh giá trên validation set
        preds = model.predict(x_val)
        acc = accuracy_score(y_val, preds)
        
        # Log metrics lên W&B
        wandb.log({"validation accuracy": acc})

wandb.agent(sweep_id, function=train, count=200)
# 1. Lấy config tốt nhất từ W&B
ENTITY   = "thanvinh164-vinh"       # thường là username hoặc team
PROJECT  = "attendance_face_recognition"
SWEEP_ID = "epiwaxnc"
api = wandb.Api()

runs = api.runs(f"{ENTITY}/{PROJECT}", {"sweep": SWEEP_ID})
best_run = max(runs, key=lambda r: r.summary.get("validation accuracy", 0.0))
best_cfg = best_run.config

print("Best hyperparameters:", best_cfg)

hyper_params = {
    k: best_cfg[k] for k in [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap"
    ]
}

print("Hyperparameters:", hyper_params)

model = RandomForestClassifier(
    **hyper_params,
    random_state=42,
    n_jobs=-1
)

# Train trên toàn bộ tập train
model.fit(x_train, y_train)

y_pred = model.predict(x_val)

print(classification_report(y_val,y_pred))

# types and names of the artifacts
artifact_type = "inference_artifact"
artifact_model = "model_export"
logger.info("Dumping the artifacts to disk")
# Save the model using joblib
joblib.dump(model, artifact_model)

# Model artifact
artifact = wandb.Artifact(artifact_model,
                          type=artifact_type,
                          description="Best Random Forest model"
                          )

logger.info("Logging model artifact")
artifact.add_file(artifact_model)
run.log_artifact(artifact)

run.finish() 


