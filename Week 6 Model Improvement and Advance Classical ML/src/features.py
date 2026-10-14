import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
  """Adds custom engineered features to the dataset."""
  df = df.copy()

  # Create an engagement score combining study hours and assignments completed
  if "study_hours_per_week" in df.columns and "assignments_completed" in df:
    df["engagement_score"] = df["study_hours_per_week"] * (
        df["assignments_completed"] / 10.0
    )

  return df