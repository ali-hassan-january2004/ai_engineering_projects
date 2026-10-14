import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features, categorical_features):
  """Builds a robust preprocessing pipeline to handle missing values,

  scaling, and categorical encoding.
  """
  numeric_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="median")),
          ("scaler", StandardScaler()),
      ]
  )

  categorical_transformer = Pipeline(
      steps=[
          ("imputer", SimpleImputer(strategy="most_frequent")),
          (
              "encoder",
              OneHotEncoder(handle_unknown="ignore", drop="first"),
          ),
      ]
  )

  preprocessor = ColumnTransformer(
      transformers=[
          ("num", numeric_transformer, numeric_features),
          ("cat", categorical_transformer, categorical_features),
      ]
  )

  return preprocessor