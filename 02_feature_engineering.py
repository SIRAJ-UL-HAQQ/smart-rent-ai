import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# Function to extract keyword amenity flags from raw rental text descriptions:
# Features to extract: 'has_laundry', 'has_parking', 'pet_friendly', 'has_balcony', 'hardwood_floors'
# Return the modified dataframe with new binary columns (1 or 0).
def extract_amenity_flags(df, text_column):
    df["has_laundry"] = df[text_column].str.contains("laundry", case=False, na=False).astype(int)
    df["has_parking"] = df[text_column].str.contains("parking", case=False, na=False).astype(int)
    df["pet_friendly"] = df[text_column].str.contains("pet", case=False, na=False).astype(int)
    df["has_balcony"] = df[text_column].str.contains("balcony", case=False, na=False).astype(int)
    df["hardwood_floors"] = df[text_column].str.contains("hardwood", case=False, na=False).astype(int)
    return df


# Complete preprocessing script with train-test split first and leakage-safe TF-IDF usage.
def build_preprocessor(df: pd.DataFrame):
    df = extract_amenity_flags(df.copy(), text_column="description")

    # 1) Split first: this is the key leakage guardrail.
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    numeric_features = ["beds", "baths", "sqft", "latitude", "longitude"]
    categorical_features = ["neighborhood"]
    binary_amenity_features = [
        "has_laundry",
        "has_parking",
        "pet_friendly",
        "has_balcony",
        "hardwood_floors",
    ]
    text_feature = "description"

    # Text transformer: TF-IDF with n-grams and max_features limit.
    text_transformer = Pipeline(
        steps=[
            (
                "select_text",
                FunctionTransformer(lambda x: x.squeeze(axis=1), validate=False),
            ),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=100,
                    stop_words="english",
                    sublinear_tf=True,
                ),
            ),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # IMPORTANT: column transformer is built after the train/test split.
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, [text_feature]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", numeric_transformer, numeric_features),
            ("binary_amenity", "passthrough", binary_amenity_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # 2) Fit only on X_train and transform X_train.
    X_train_preprocessed = preprocessor.fit_transform(X_train)

    # 3) Transform X_test only. Never fit on X_test.
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test


if __name__ == "__main__":
    synthetic_df = pd.read_csv("synthetic_rentals.csv")
    X_train_preprocessed, X_test_preprocessed, y_train, y_test = build_preprocessor(synthetic_df)

    print("X_train_preprocessed shape:", X_train_preprocessed.shape)
    print("X_test_preprocessed shape:", X_test_preprocessed.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
