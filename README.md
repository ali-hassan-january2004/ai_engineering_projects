# AI Engineering Projects

Weekly AI/ML exercises and projects, with notebooks and reusable Python scripts.

## Set up on Windows

Use Python 3.13 for this environment. If it is not installed, install it with the
Python install manager:

```powershell
py install 3.13
```

Open this `ai_engineering_projects` folder in VS Code, then run these commands
in its PowerShell terminal:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter and
select the `.venv` environment as the kernel when opening a notebook. The
requirements file includes the shared notebook/ML dependencies and XGBoost
used by Week 6.

## Run the project scripts

Run these examples from this folder, using the environment's Python:

```powershell
.\.venv\Scripts\python.exe ".\Week 3 Data Preprocessing\Mini_Project_Week_3\src\preprocessing.py"
.\.venv\Scripts\python.exe ".\Week 4 Supervised ML\Mini_Project_Week_4\src\train.py"
.\.venv\Scripts\python.exe ".\Week 5 UnSupervised ML\main.py"
.\.venv\Scripts\python.exe ".\Week 6 Model Improvement and Advance Classical ML\src\train.py"
```

Open any `.ipynb` notebook in VS Code and run its cells with the `.venv`
kernel.

## Adding projects

Create a clearly named folder for each new project. Keep reusable code in
`src/`, notebooks in `notebooks/`, datasets in `data/`, and results in
`reports/`. Reuse this environment for the current curriculum; add any new
dependencies to `requirements.txt` and reinstall with the command above. If a
project later needs conflicting package versions, give that project its own
virtual environment and requirements file.
