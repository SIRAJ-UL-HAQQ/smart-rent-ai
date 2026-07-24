import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from preprocessing_utils import squeeze_text_column


# -----------------------------
# Reuse the same feature engineering logic from 02_feature_engineering.py
# -----------------------------
def extract_amenity_flags(df, text_column):
    df["has_laundry"] = df[text_column].str.contains("laundry", case=False, na=False).astype(int)
    df["has_parking"] = df[text_column].str.contains("parking", case=False, na=False).astype(int)
    df["pet_friendly"] = df[text_column].str.contains("pet", case=False, na=False).astype(int)
    df["has_balcony"] = df[text_column].str.contains("balcony", case=False, na=False).astype(int)
    df["hardwood_floors"] = df[text_column].str.contains("hardwood", case=False, na=False).astype(int)
    return df


def build_preprocessor(df: pd.DataFrame):
    df = extract_amenity_flags(df.copy(), text_column="description")

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

    text_transformer = Pipeline(
        steps=[
            (
                "select_text",
                FunctionTransformer(squeeze_text_column, validate=False),
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

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor, X_train_preprocessed, X_test_preprocessed


# -----------------------------
# Model evaluation helpers
# -----------------------------
def compute_metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_markdown_table(results):
    headers = ["Model", "RMSE", "MAE", "R2"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for model_name, metrics in results.items():
        print(
            f"| {model_name} | {metrics['RMSE']:.3f} | {metrics['MAE']:.3f} | {metrics['R2']:.4f} |"
        )


# -----------------------------
# Run pipeline
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("synthetic_rentals.csv")
    X_train, X_test, y_train, y_test, preprocessor, X_train_preprocessed, X_test_preprocessed = build_preprocessor(df)

    # Baseline model: Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_preprocessed, y_train)
    ridge_pred = ridge.predict(X_test_preprocessed)
    ridge_metrics = compute_metrics(y_test, ridge_pred)

    # Save the fitted preprocessor + fitted Ridge model together as a single reusable artifact.
    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("ridge", ridge),
    ])

    # Fit the pipeline on the raw training frame so the artifact is ready for future inference.
    model_pipeline.fit(X_train, y_train)
    dump(model_pipeline, "rental_ridge_model.joblib")
    print("Saved fitted pipeline to rental_ridge_model.joblib")

    # Extract feature names from the fitted ColumnTransformer.
    text_vectorizer = preprocessor.named_transformers_["text"].named_steps["tfidf"]
    text_feature_names = text_vectorizer.get_feature_names_out()
    categorical_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(
        input_features=["neighborhood"]
    )

    numeric_features = ["beds", "baths", "sqft", "latitude", "longitude"]
    binary_amenity_features = [
        "has_laundry",
        "has_parking",
        "pet_friendly",
        "has_balcony",
        "hardwood_floors",
    ]

    feature_names = (
        list(text_feature_names)
        + list(categorical_feature_names)
        + numeric_features
        + binary_amenity_features
    )

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": ridge.coef_,
        }
    )

    positive_features = coef_df.sort_values("coefficient", ascending=False).head(10)
    negative_features = coef_df.sort_values("coefficient", ascending=True).head(5)
    top_features = pd.concat([positive_features, negative_features], ignore_index=True)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_features,
        x="coefficient",
        y="feature",
        hue="coefficient",
        palette="coolwarm",
        dodge=False,
    )
    plt.axvline(0, color="black", linewidth=1, linestyle="--")
    plt.title("Top Ridge Coefficients for Rental Price")
    plt.xlabel("Ridge Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("ridge_coefficients.png", dpi=150)
    plt.close()

    # Model 2: LightGBM Regressor
    try:
        from lightgbm import LGBMRegressor

        lgbm = LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        lgbm.fit(X_train_preprocessed, y_train)
        lgbm_pred = lgbm.predict(X_test_preprocessed)
        lgbm_metrics = compute_metrics(y_test, lgbm_pred)
    except Exception as exc:
        print(f"LightGBM could not be imported or trained: {exc}")
        lgbm_metrics = None

    results = {
        "Ridge Regression": ridge_metrics,
    }
    if lgbm_metrics is not None:
        results["LightGBM Regressor"] = lgbm_metrics

    print_markdown_table(results)

    # Pick the best model by lowest RMSE
    if lgbm_metrics is not None:
        best_model_name = min(
            results,
            key=lambda name: results[name]["RMSE"],
        )
        best_predictions = ridge_pred if best_model_name == "Ridge Regression" else lgbm_pred
    else:
        best_model_name = "Ridge Regression"
        best_predictions = ridge_pred

    # Scatter plot for best model
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=y_test,
        y=best_predictions,
        alpha=0.7,
        color="#4C78A8",
    )
    min_val = min(y_test.min(), best_predictions.min())
    max_val = max(y_test.max(), best_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="red", linewidth=1)
    plt.title(f"Actual vs. Predicted Prices - {best_model_name}")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=150)
    plt.close()
