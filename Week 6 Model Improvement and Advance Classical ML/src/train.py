import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from preprocessing import build_preprocessor


def train_and_evaluate():
  # Load dataset
  df = pd.read_csv("../data/student_performance.csv")

  X = df.drop(columns=["student_id", "passed"])
  y = df["passed"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  numeric_features = [
      "study_hours_per_week",
      "attendance_rate",
      "previous_gpa",
      "assignments_completed",
      "midterm_score",
  ]
  categorical_features = ["extracurricular"]

  preprocessor = build_preprocessor(numeric_features, categorical_features)

  models = {
      "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
      "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
      "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
      "Gradient Boosting": GradientBoostingClassifier(random_state=42),
  }

  for name, model in models.items():
    pipe = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model)]
    )
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    score = f1_score(y_test, preds)
    print(f"Model: {name} | Test F1 Score: {score:.4f}")


if __name__ == "__main__":
  train_and_evaluate()