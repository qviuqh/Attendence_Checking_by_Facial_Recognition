### Welcome to my project

# 📸 Attendance Checking with Facial Recognition - MLOps Project

## 📚 Contents

- [🔰 Introduction](#-introduction)
- [🏗️ Architecture](#-architecture)
- [📁 Folder Structure](#-folder-structure)
- [⚙️ Installation](#-installation)
  - [1. Clone the repository](#1-clone-the-repository)
  - [2. Create and activate virtual environment](#2-create-and-activate-virtual-environment)
  - [3. Install dependencies](#3-install-dependencies)
  - [4. Configure Weights & Biases](#4-configure-weights--biases)
- [🖼️ Example UI or Output](#-example-ui-or-output)
- [⚙️ CI/CD Pipeline](#-cicd-pipeline)
- [🧭 Future Roadmap](#-future-roadmap)
- [📄 References](#-references)

## 🔰 Introduction

This project aims to develop a facial recognition-based attendance system to streamline student tracking in classrooms. By leveraging Machine Learning and MLOps practices, the system automates attendance, improving accuracy and operational efficiency. It provides a scalable, maintainable solution for educational institutions looking to replace traditional manual methods.

Manual attendance tracking in classrooms is time-consuming, error-prone, and susceptible to fraud. These issues are amplified in large-scale classes, where administrative tasks can hinder instructional time and classroom efficiency. Additionally, the lack of automation makes it difficult to scale or integrate with modern digital education infrastructures.

To address these challenges, the project proposes an end-to-end facial recognition attendance system. The solution integrates real-time video input with trained face recognition models for automated student identification and attendance logging. The system incorporates an MLOps pipeline, ensuring automation in data versioning, model training, continuous integration/deployment (CI/CD), monitoring, and retraining.

The main objective of this project is to automate the attendance process, saving time for both instructors and students. It aims to:
- Minimize errors and prevent cheating by using facial recognition instead of roll calls or sign-in sheets.
- Update and store attendance data in real-time, enabling easy tracking, reporting, and management.
- Enhance the professionalism and modernization of classroom management in educational institutions.

---

## 🏗️ Architecture

_Add your architecture diagram below:_

![System Architecture](path/to/your/architecture_diagram.png)

**Components Overview:**
- Face capture and preprocessing
- Embedding generation (e.g., FaceNet)
- Classifier for student ID prediction
- FastAPI server with endpoints
- Weights & Biases for model tracking
- CI/CD pipeline (GitHub Actions, etc.)

---

# Attendance Checking by Facial Recognition

This project develops a facial recognition-based attendance system, integrating face detection, attendance management, and model training.

## 📁 Folder Structure

```bash
Attendance_Checking_by_Facial_Recognition/
├── .github/
│   └── workflows/                    # GitHub Actions CI/CD workflows.
├── UI/                                # UI source code and components.
│   ├── icon/                          # UI icons.
│   ├── src/                           # UI source code.
│   ├── api_client.py                  # Backend API client.
│   ├── attendance_manager.py          # Manages attendance logic.
│   ├── dialogs.py                     # UI dialog windows.
│   ├── face_detector.py               # Face detection logic.
│   ├── main_window.py                 # Main UI window.
│   ├── test.ipynb                     # Jupyter notebook for UI testing.
│   └── ui.py                          # Core UI layout and events.
├── api/                                # FastAPI source code.
│   └── main.py                        # API endpoints.
├── data/                               # Input images, embeddings, datasets.
│   ├── images/                        # Face images used for training/testing.
│   └── embeddings/                    # Face embeddings for recognition.
├── model/                              # Trained model artifacts.
│   ├── model_export/                  # Folder for exported models.
│   └── retrain/                       # Model retraining scripts.
│       ├── 2_Data_segregation.py      # Data segregation for retraining.
│       ├── 3_Training_model.py        # Model training script.
│       └── 4_Testing.py               # Model testing script.
├── notebooks/                          # Jupyter notebooks for training, logging, and testing.
│   ├── 1. ETL_process.ipynb            # ETL process for data preparation.
│   ├── 2. Data_segregation.ipynb       # Data segregation for training/testing.
│   ├── 3. Training_model.ipynb         # Model training.
│   └── 4. Testing.ipynb                # Model testing.
├── artifacts/                         # Stores model artifacts and other resources.
├── log.csv                             # Attendance logs.
├── requirements.txt                   # Project dependencies.
├── Procfile                            # For Heroku deployment (optional).
└── README.md                           # Project overview and setup instructions.

```
---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

### 2. Create and activate virtual environment

```bash
python -m venv .env
# On macOS/Linux:
source .env/bin/activate
# On Windows:
.env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Weights & Biases

Create a `.env` file and add your W&B API key:

```bash
WANDB_API_KEY=your_wandb_key_here
```

---

## 🖼️ Example UI or Output

### 🧑‍💻 How to Use the Application

This section explains the user flow and interactions with the desktop attendance app:

1. **Startup Interface**:
   - When the application launches, it shows **3 buttons**:
     - `Take Attendance`
     - `Register New Face`
     - `Show Attendance List`

2. **Show Attendance List**:
   - When clicked, it opens a list displaying:
     - Student **Name**
     - **ID**
     - **Present status**
   - This button toggles to **Close Attendance List** when opened.

3. **Register New Face**:
   - Prompts the user to **input a video** of a new student following a required format.
   - The system will extract facial features and associate them with a student ID.

4. **Take Attendance**:
   - Starts the webcam for real-time recognition.
   - Three new buttons appear:
     - ✅ `Confirm` – confirms detected faces and updates the attendance list.
     - 🔁 `Retry Wrong Face` – allows students to input their **Student ID** if misclassified, then re-checks.
     - 🔙 `Back` – stops the camera and returns to the initial 3-button interface. The extra buttons (`Confirm`, `Retry`, `Back`) disappear.

5. **Flow Logic**:
   - After each confirmation, the `Take Attendance` process continues.
   - Exiting via `Back` will cleanly shut the camera and return to the main screen.

---

### 📷 Sample Screenshots

- ![Startup Interface](How_to_use_image/Ui1.png)
- ![Attendance List View](How_to_use_image/Ui2.png)
- ![Take Attendance ](How_to_use_image/Ui3.png)
- ![Retry Input](How_to_use_image/Ui4.png)


## ⚙️ CI/CD Pipeline

_Describe your automation here:_

This project uses **GitHub Actions** to: automatically retrain a machine learning model whenever new video data is uploaded. The process includes preprocessing video data, generating embeddings, and uploading both embeddings and the updated model to Weights & Biases (WandB).

Key Features:
- Fully automated retraining via GitHub Actions triggered by a repository_dispatch event from a backend system.
- Sequential execution of data segregation, training, and testing scripts.
- Secure WandB integration using GitHub Secrets.
- Resulting model artifacts (e.g., .pkl files) are logged and stored in WandB.

_Workflow file:_
```bash
.github/workflows/train_and_deploy.yaml
```

---

## 🧭 Future Roadmap

- [ ] Dockerize API and desktop app
- [ ] Add student registration via UI
- [ ] Add real-time attendance dashboard
- [ ] Deploy API to cloud (Render/AWS/GCP)

---

## 📄 References

- [FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Weights & Biases](https://wandb.ai)
- Inspired by: [Credit Risk MLOps](https://github.com/Morsinaldo/credit_risk_mlops)
