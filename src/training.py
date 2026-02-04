"""
Training module.

- loads cleaned data.csv
- splits into train/test
- builds preprocessor (imputer+scaler / imputer+ohe)
- trains multiple models with GridSearchCV
- saves per-model best and overall best model
- saves feature column list to models/feature_columns.json
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

from src.config import (
    CLEANED_DATA_PATH, 
    MODEL_DIR, 
    BEST_MODEL_PATH, 
    FEATURE_PATH, 
    CV_FOLD, 
    SCORING, 
    N_JOBS
)
from src.preprocessing import load_data, split_data, preprocessor   


def get_models_and_params() -> List[Tuple[str, object, Dict[str, List]]]:
    """
    Returns list of (name, estimator, param_grid) tuples for GridSearchCV
    """
    models_and_params = []
    
    # 1. Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg_params = {
        "clf__C": [0.01, 0.1, 1.0], 
        "clf__penalty": ["l2"], 
        "clf__solver": ["lbfgs"]
    }
    models_and_params.append(("log_reg", log_reg, log_reg_params))
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt_params = {
        "clf__max_depth": [None, 5, 10, 15],
        "clf__min_samples_split": [2, 5, 10],
        "clf__criterion": ["gini", "entropy"]
    }
    models_and_params.append(("decision_tree", dt, dt_params))
    
    # 3. Random Forest
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        "clf__n_estimators": [100, 200], 
        "clf__max_depth": [None, 5, 10]
    }
    models_and_params.append(("random_forest", rf, rf_params))
    
    # 4. K-Nearest Neighbors
    knn = KNeighborsClassifier()
    knn_params = {
        "clf__n_neighbors": [3, 5, 7, 9],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"]
    }
    models_and_params.append(("knn", knn, knn_params))
    
    # 5. Support Vector Machine
    svc = SVC(probability=True)
    svc_params = {
        "clf__C": [0.1, 1.0], 
        "clf__kernel": ["rbf", "linear"]
    }
    models_and_params.append(("svc", svc, svc_params))
    
    # 6. Gradient Boosting
    gb = GradientBoostingClassifier(random_state=42)
    gb_params = {
        "clf__n_estimators": [100, 200], 
        "clf__learning_rate": [0.05, 0.1], 
        "clf__max_depth": [3, 5]
    }
    models_and_params.append(("gradient_boosting", gb, gb_params))
    
    
    return models_and_params


def train_and_select_model() -> pd.DataFrame:
    """
    Main training function:
    1. Load cleaned data
    2. Split train/test
    3. Build preprocessor
    4. Train all models with GridSearchCV
    5. Save best models
    6. Return results DataFrame
    """
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    
    print("STARTING MODEL TRAINING WITH GRIDSEARCH")
    
    
    # Step 1: Load data
    print(f"\nLoading pre-cleaned data -> {CLEANED_DATA_PATH}")
    df = load_data(CLEANED_DATA_PATH)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Step 2: Split data
    # Save feature columns (before any accidental modifications) and split
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Save the training feature list to disk for runtime alignment
    feature_cols = X_train.columns.tolist()
    with open(FEATURE_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)
    print(f"Saved feature list ({len(feature_cols)}) to {FEATURE_PATH}")
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # Step 3: Build preprocessor
    preprocessor_pipeline = preprocessor(X_train)
    print("Preprocessor built.")
    
    # Step 4: Get models
    models_and_params = get_models_and_params()
    
    best_overall = None
    best_score = -np.inf
    best_name = None
    results = []
    
    # Step 5: Train each model
    print(f"\nTraining {len(models_and_params)} models...")
    
    for name, estimator, param_grid in models_and_params:
        print(f"\n--- Training {name} ---")
        
        # Create pipeline: preprocessor + classifier
        pipeline = Pipeline([
            ("preprocess", preprocessor_pipeline), 
            ("clf", estimator)
        ])
        
        # GridSearchCV
        grid = GridSearchCV(
            estimator=pipeline, 
            param_grid=param_grid, 
            cv=CV_FOLD, 
            scoring=SCORING, 
            n_jobs=N_JOBS, 
            verbose=1
        )
        grid.fit(X_train, y_train)
        
        # Best model from grid search
        best = grid.best_estimator_
        cv_score = grid.best_score_
        
        # Predictions on test set
        y_pred = best.predict(X_test)
        test_f1 = metrics.f1_score(y_test, y_pred, average="macro")
        
        print(f"{name} best params: {grid.best_params_}")
        print(f"{name} CV {SCORING}: {cv_score:.4f}")
        print(f"{name} Test F1_macro: {test_f1:.4f}")
        print(metrics.classification_report(y_test, y_pred, zero_division= 0))
        
        # Save individual model
        model_path = MODEL_DIR / f"{name}_best_model.joblib"
        joblib.dump(best, model_path)
        print(f"Saved {name} -> {model_path}")
        
        # Store results
        results.append({
            "model": name, 
            "best_params": grid.best_params_, 
            "cv_score": cv_score, 
            "test_f1_macro": test_f1
        })
        
        # Track overall best
        if test_f1 > best_score:
            best_score = test_f1
            best_overall = best
            best_name = name
    
    # Step 6: Save overall best model
    if best_overall is not None:
        joblib.dump(best_overall, BEST_MODEL_PATH)
        print(f"\nOverall best model: {best_name} (Test F1_macro: {best_score:.4f}) -> {BEST_MODEL_PATH}")
    else:
        print("No model trained successfully.")
    
    
    return  pd.DataFrame(results)
    


if __name__ == "__main__":
    df_results = train_and_select_model()
    print("\nSummary:\n", df_results)
