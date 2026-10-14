# Week 6 Classical ML Experiment

A reproducible experiment comparing multiple classical machine learning models on a student performance tabular dataset featuring missing values and realistic noise.

## Project Structure
* `data/`: Houses the raw `student_performance.csv` dataset.
* `src/`: Contains reusable Python modules for preprocessing (`preprocessing.py`) and training (`train.py`).
* `notebooks/`: Interactive notebooks for baselines, modeling, tuning, and error analysis.
* `reports/`: Documentation and model comparison summaries.

## Setup Instructions (VS Code)

1. **Open the project folder** in VS Code.
2. **Create and activate a Python virtual environment** in your VS Code terminal:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows (Command Prompt): venv\Scripts\activate
   # On Windows (PowerShell): .\venv\Scripts\Activate.ps1