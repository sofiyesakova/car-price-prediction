"""
Klassifikation der Fahrzeugpreiskategorie

Dieses Skript ist die .py-Version des Notebooks:
Klassifikation_PreisKategorie.ipynb

Ziel:
- Fahrzeugpreise werden nicht als exakte Zahl vorhergesagt.
- Stattdessen werden Fahrzeuge in drei Preisgruppen klassifiziert:
  Günstig, Mittel, Teuer.

Eingabedatei:
- Standard: data/processed/cleaned_data.csv
- Die Datei muss eine Spalte "Preis_Euro" und die unten definierten Feature-Spalten enthalten.

Ausführen:
    python Klassifikation_PreisKategorie.py

Optional mit eigenem Pfad:
    python Klassifikation_PreisKategorie.py --data-path data/processed/cleaned_data.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_PRICE_COLUMN = "Preis_Euro"
TARGET_CLASS_COLUMN = "Preis_Kategorie"

FEATURES = [
    "Verkaufszahl",
    "Hubraum_L",
    "Kundenzufriedenheit",
    "Jahr",
    "Monat",
    "Wochentag",
    "Marke",
    "Modell",
    "Kraftstoff",
    "Getriebe",
    "Bundesland",
]

NUMERIC_FEATURES = [
    "Verkaufszahl",
    "Hubraum_L",
    "Kundenzufriedenheit",
    "Jahr",
    "Monat",
]

CATEGORICAL_FEATURES = [
    "Marke",
    "Modell",
    "Kraftstoff",
    "Getriebe",
    "Bundesland",
    "Wochentag",
]

CLASS_LABELS = ["Günstig", "Mittel", "Teuer"]


def load_data(data_path: str | Path) -> pd.DataFrame:
    """CSV-Datei laden."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Die Datei wurde nicht gefunden: {data_path}\n"
            "Bitte den richtigen Pfad mit --data-path angeben."
        )

    df = pd.read_csv(data_path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    """Prüfen, ob alle benötigten Spalten vorhanden sind."""
    required_columns = [TARGET_PRICE_COLUMN] + FEATURES
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "Diese benötigten Spalten fehlen in der CSV-Datei:\n"
            + "\n".join(f"- {column}" for column in missing_columns)
        )


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preis in drei Klassen einteilen.

    pd.qcut teilt die Daten so ein, dass die drei Gruppen möglichst ähnlich groß sind.
    """
    df = df.copy()

    df[TARGET_CLASS_COLUMN] = pd.qcut(
        df[TARGET_PRICE_COLUMN],
        q=3,
        labels=CLASS_LABELS,
    )

    return df


def build_preprocessor() -> ColumnTransformer:
    """Numerische Daten skalieren und kategorische Daten per One-Hot-Encoding umwandeln."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_logistic_regression_model(preprocessor: ColumnTransformer) -> Pipeline:
    """Pipeline für logistische Regression bauen."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest_model(preprocessor: ColumnTransformer) -> Pipeline:
    """Pipeline für Random Forest bauen."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(model_name: str, y_test: pd.Series, predictions: pd.Series) -> dict:
    """Accuracy, F1-Scores und Classification Report ausgeben."""
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")
    weighted_f1 = f1_score(y_test, predictions, average="weighted")

    print(f"\n{'=' * 60}")
    print(model_name)
    print(f"{'=' * 60}")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1-Score:", round(macro_f1, 4))
    print("Weighted F1-Score:", round(weighted_f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return {
        "Modell": model_name,
        "Accuracy": accuracy,
        "Macro F1-Score": macro_f1,
        "Weighted F1-Score": weighted_f1,
    }


def save_confusion_matrix(y_test: pd.Series, predictions: pd.Series, output_dir: Path) -> None:
    """Confusion Matrix als Bild speichern."""
    cm = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()

    output_path = output_dir / "random_forest_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Confusion Matrix gespeichert: {output_path}")


def get_feature_importance(rf_model: Pipeline) -> pd.DataFrame:
    """Feature Importance aus dem Random-Forest-Modell extrahieren."""
    rf_classifier = rf_model.named_steps["classifier"]
    feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "Merkmal": feature_names,
            "Wichtigkeit": rf_classifier.feature_importances_,
        }
    )

    importance_df = importance_df.sort_values(
        by="Wichtigkeit",
        ascending=False,
    )

    return importance_df


def save_feature_importance_plot(importance_df: pd.DataFrame, output_dir: Path) -> None:
    """Top 15 Feature Importances als Bild speichern."""
    top_features = importance_df.head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_features,
        x="Wichtigkeit",
        y="Merkmal",
    )

    plt.title("Top 15 wichtigste Merkmale")
    plt.xlabel("Wichtigkeit")
    plt.ylabel("Merkmal")
    plt.tight_layout()

    output_path = output_dir / "top_15_feature_importances.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Feature-Importance-Plot gespeichert: {output_path}")


def save_model_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    """Accuracy-Vergleich der Modelle als Bild speichern."""
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=comparison_df,
        x="Modell",
        y="Accuracy",
    )

    plt.title("Vergleich der Modellgenauigkeit")
    plt.ylabel("Accuracy")
    plt.xlabel("Modell")
    plt.ylim(0, 1)
    plt.tight_layout()

    output_path = output_dir / "modellvergleich_accuracy.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Modellvergleich gespeichert: {output_path}")


def run_analysis(data_path: str | Path, output_dir: str | Path) -> None:
    """Komplette Klassifikationsanalyse ausführen."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)
    validate_columns(df)

    print("Daten geladen.")
    print("DataFrame Shape:", df.shape)
    print("\nSpalten:")
    print(df.columns.tolist())

    df = create_target_variable(df)

    print("\nVerteilung der Preis-Kategorien:")
    print(df[TARGET_CLASS_COLUMN].value_counts())

    X = df[FEATURES]
    y = df[TARGET_CLASS_COLUMN]

    print("\nX shape:", X.shape)
    print("y shape:", y.shape)

    print("\nNumerische Features:")
    print(NUMERIC_FEATURES)

    print("\nKategorische Features:")
    print(CATEGORICAL_FEATURES)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain/Test Split:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # Wichtig: Für jedes Modell einen eigenen Preprocessor erstellen,
    # damit beide Pipelines sauber unabhängig voneinander trainiert werden.
    log_reg_model = build_logistic_regression_model(build_preprocessor())
    rf_model = build_random_forest_model(build_preprocessor())

    log_reg_model.fit(X_train, y_train)
    log_reg_pred = log_reg_model.predict(X_test)
    print("\nLogistic Regression training completed.")

    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    print("Random Forest training completed.")

    results = [
        evaluate_model("Logistische Regression", y_test, log_reg_pred),
        evaluate_model("Random Forest", y_test, rf_pred),
    ]

    comparison_df = pd.DataFrame(results)
    comparison_path = output_dir / "modellvergleich.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nModellvergleich gespeichert: {comparison_path}")
    print(comparison_df)

    save_confusion_matrix(y_test, rf_pred, output_dir)

    importance_df = get_feature_importance(rf_model)
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature Importance gespeichert: {importance_path}")
    print("\nTop 15 wichtigste Merkmale:")
    print(importance_df.head(15))

    save_feature_importance_plot(importance_df, output_dir)
    save_model_comparison_plot(comparison_df, output_dir)

    print("\nCross Validation für Random Forest:")
    cv_scores = cross_val_score(
        rf_model,
        X,
        y,
        cv=5,
        scoring="accuracy",
    )

    print("Cross Validation Scores:")
    print(cv_scores)
    print("\nDurchschnittliche Accuracy:")
    print(cv_scores.mean())
    print("\nStandardabweichung:")
    print(cv_scores.std())

    print("\nInterpretation:")
    print(
        "In dieser Analyse wurden logistische Regression und Random Forest verglichen. "
        "Das Ziel ist, Fahrzeuge anhand ihrer Merkmale in die Preisgruppen "
        "'Günstig', 'Mittel' und 'Teuer' einzuteilen. "
        "Random Forest kann nichtlineare Zusammenhänge besser erkennen, "
        "während logistische Regression einfacher und besser interpretierbar ist."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Klassifikation von Fahrzeugen in Preis-Kategorien."
    )

    parser.add_argument(
        "--data-path",
        default="data/processed/cleaned_data.csv",
        help="Pfad zur CSV-Datei. Standard: data/processed/cleaned_data.csv",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/klassifikation_preiskategorie",
        help="Ordner für Ergebnisse und Plots.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_analysis(
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
