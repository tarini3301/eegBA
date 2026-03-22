"""
Brain Age Prediction — Multi-Model Engine
==========================================
Trains 9 ML models on synthetic DS003775-style data and provides
SHAP-based explainability. Users can select any model or use the
ensemble for the most robust prediction.

Models:
  1. Random Forest
  2. Gradient Boosting (XGBoost-style via HistGradientBoosting)
  3. Support Vector Regression (SVR with RBF kernel)
  4. Lasso Regression (L1-regularized linear model)
  5. Ridge Regression (L2-regularized linear model)
  6. ElasticNet Regression (combined L1/L2 regularization)
  7. K-Nearest Neighbors
  8. Neural Network (MLP with 3 hidden layers)
  9. Bayesian Ridge Regression
  10. Ensemble (weighted average of all 9 models)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import shap
import lime
import lime.lime_tabular
import joblib
import os

# ─────────────────────────────────────────────
# Feature definitions with aging trajectories
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    f"{region}_{band}_Power" for region in ["Frontal", "Central", "Temporal", "Parietal", "Occipital"] for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
]

FEATURE_DISPLAY_NAMES = {
    f"{region}_{band}_Power": f"{region} {band} Power (µV²)"
    for region in ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]
    for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
}

FEATURE_UNITS = {
    name: "µV²" for name in FEATURE_NAMES
}

# ─────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────

MODEL_INFO = {
    "random_forest": {
        "name": "Random Forest",
        "short": "RF",
        "description": "Ensemble of decision trees with bootstrap aggregation. Robust, handles non-linear relationships, and naturally resistant to overfitting.",
        "type": "tree",
    },
    "gradient_boosting": {
        "name": "Gradient Boosting",
        "short": "GB",
        "description": "Sequential ensemble that builds trees to correct errors of previous ones. Often the top performer for tabular data.",
        "type": "tree",
    },
    "svr": {
        "name": "Support Vector Regression",
        "short": "SVR",
        "description": "Finds optimal hyperplane in high-dimensional space using RBF kernel. Excellent for small-medium datasets with complex boundaries.",
        "type": "kernel",
    },
    "lasso": {
        "name": "Lasso Regression",
        "short": "Lasso",
        "description": "Linear model with L1 regularization that performs automatic feature selection by driving irrelevant coefficients to zero.",
        "type": "linear",
    },
    "ridge": {
        "name": "Ridge Regression",
        "short": "Ridge",
        "description": "Linear model with L2 regularization to prevent overfitting by penalizing large coefficients.",
        "type": "linear",
    },
    "elastic_net": {
        "name": "ElasticNet Regression",
        "short": "ENet",
        "description": "Combines L1 and L2 regularizations, balancing feature selection and coefficient shrinkage.",
        "type": "linear",
    },
    "knn": {
        "name": "K-Nearest Neighbors",
        "short": "KNN",
        "description": "Instance-based learning that predicts age based on the closest subjects in the feature space.",
        "type": "distance",
    },
    "mlp": {
        "name": "Neural Network (Deep Learning)",
        "short": "MLP",
        "description": "Multi-Layer Perceptron with 3 deep hidden layers. Captures complex, high-order non-linear interactions between EEG bands.",
        "type": "kernel",
    },
    "bayesian_ridge": {
        "name": "Bayesian Ridge",
        "short": "BRidge",
        "description": "A probabilistic approach to linear regression that provides uncertainty estimates and robustness out-of-the-box.",
        "type": "linear",
    },
    "ensemble": {
        "name": "Ensemble (All Models)",
        "short": "ENS",
        "description": "Weighted average of all individual models, with weights proportional to their R² scores. Most robust prediction.",
        "type": "ensemble",
    },
}


def generate_synthetic_dataset(n_subjects=111, random_state=42):
    """
    Generate a synthetic dataset that mimics DS003775 resting-state EEG patterns.
    """
    rng = np.random.RandomState(random_state)

    ages_young  = rng.uniform(19, 35, size=int(n_subjects * 0.50))
    ages_middle = rng.uniform(35, 55, size=int(n_subjects * 0.30))
    ages_old    = rng.uniform(55, 65, size=n_subjects - len(ages_young) - len(ages_middle))
    ages = np.concatenate([ages_young, ages_middle, ages_old])
    rng.shuffle(ages)

    age_norm = (ages - 19) / 46.0  # normalize age 19-65

    data_dict = {"chronological_age": np.round(ages, 1)}
    
    for region in ["Frontal", "Central", "Temporal", "Parietal", "Occipital"]:
        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            feature = f"{region}_{band}_Power"
            
            if band == "Delta":
                base = 25.0
                age_effect = 10.0 * age_norm
            elif band == "Theta":
                base = 15.0
                age_effect = 5.0 * age_norm
            elif band == "Alpha":
                base = 35.0 if region in ["Occipital", "Parietal"] else 20.0
                age_effect = -15.0 * age_norm
            elif band == "Beta":
                base = 12.0
                age_effect = -3.0 * age_norm
            else: # Gamma
                base = 5.0
                age_effect = -1.0 * age_norm
                
            if region == "Frontal" and band in ["Delta", "Theta"]:
                base *= 1.2
            
            val = base + age_effect + rng.normal(0, base * 0.15, n_subjects)
            data_dict[feature] = np.round(np.maximum(0.1, val), 2)
            
    genders = rng.choice(["M", "F"], size=n_subjects, p=[0.4, 0.6])
    data = pd.DataFrame(data_dict)
    data.insert(0, "gender", genders)
    data.insert(0, "subject_id", [f"sub-{i:03d}" for i in range(1, n_subjects + 1)])
    
    return data


# ─────────────────────────────────────────────
# Multi-Model Manager
# ─────────────────────────────────────────────

class BrainAgeModel:
    """
    Manages 5 ML models for brain age prediction with SHAP explainability.
    Caches to disk on first run, loads from cache on subsequent runs.
    """

    CACHE_PATH = os.path.join(os.path.dirname(__file__), "trained_models.joblib")
    SCALER_PATH = os.path.join(os.path.dirname(__file__), "trained_scaler.joblib")
    DATA_PATH = os.path.join(os.path.dirname(__file__), "synthetic_dataset.csv")

    def __init__(self):
        self.feature_names = FEATURE_NAMES
        self.feature_display_names = FEATURE_DISPLAY_NAMES
        self.models = {}        # model_key → sklearn model
        self.scores = {}        # model_key → R² on test set
        self.cv_scores = {}     # model_key → mean CV R²
        self.mae_scores = {}    # model_key → MAE on test set
        self.explainers = {}    # model_key → SHAP explainer

        if os.path.exists(self.CACHE_PATH) and os.path.exists(self.SCALER_PATH):
            self._load_models()
        else:
            self._train_all_models()

    # ─── Training ───

    def _train_all_models(self):
        print("🧠 Generating synthetic dataset (111 subjects)...")
        self.data = generate_synthetic_dataset()
        self.data.to_csv(self.DATA_PATH, index=False)

        X = self.data[self.feature_names].values
        y = self.data["chronological_age"].values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        # ── 1. Random Forest ──
        print("\n📊 Training Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=5,
            min_samples_split=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )
        self._train_and_score(rf, "random_forest", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 2. Gradient Boosting ──
        print("📊 Training Gradient Boosting...")
        gb = HistGradientBoostingRegressor(
            max_iter=100, max_depth=3,
            learning_rate=0.05, min_samples_leaf=10,
            random_state=42,
        )
        self._train_and_score(gb, "gradient_boosting", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 3. SVR ──
        print("📊 Training SVR...")
        svr = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
        self._train_and_score(svr, "svr", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 4. Lasso ──
        print("📊 Training Lasso Regression...")
        lasso = Lasso(alpha=0.5, max_iter=5000, random_state=42)
        self._train_and_score(lasso, "lasso", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 5. Ridge ──
        print("📊 Training Ridge Regression...")
        ridge = Ridge(alpha=10.0, random_state=42)
        self._train_and_score(ridge, "ridge", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 6. ElasticNet ──
        print("📊 Training ElasticNet Regression...")
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
        self._train_and_score(elastic, "elastic_net", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 7. KNN ──
        print("📊 Training K-Nearest Neighbors...")
        knn = KNeighborsRegressor(n_neighbors=7, weights='uniform')
        self._train_and_score(knn, "knn", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 8. MLP (Deep Learning) ──
        print("📊 Training Deep Neural Network (MLP)...")
        mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size='auto',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self._train_and_score(mlp, "mlp", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── 9. Bayesian Ridge ──
        print("📊 Training Bayesian Ridge Regression...")
        bridge = BayesianRidge()
        self._train_and_score(bridge, "bayesian_ridge", X_train, X_test, y_train, y_test, X_scaled, y)

        # ── Build SHAP and LIME explainers ──
        print("\n🔍 Building explainers...")
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=self.feature_names,
            class_names=["Brain Age"],
            mode="regression",
            random_state=42
        )
        
        for key, model in self.models.items():
            mtype = MODEL_INFO[key]["type"]
            if mtype == "tree":
                self.explainers[key] = shap.TreeExplainer(model)
            else:
                # Use KernelExplainer with a background sample for non-tree models
                bg = shap.sample(pd.DataFrame(X_scaled, columns=self.feature_names), 50)
                self.explainers[key] = shap.KernelExplainer(model.predict, bg)

        # ── Save ──
        cache = {
            "models": self.models,
            "scores": self.scores,
            "cv_scores": self.cv_scores,
            "mae_scores": self.mae_scores,
        }
        joblib.dump(cache, self.CACHE_PATH)
        joblib.dump(self.scaler, self.SCALER_PATH)

        self._print_summary()

    def _train_and_score(self, model, key, X_train, X_test, y_train, y_test, X_full, y_full):
        """Train model, compute scores, retrain on full data."""
        model.fit(X_train, y_train)

        # Test R²
        r2 = model.score(X_test, y_test)

        # MAE
        preds = model.predict(X_test)
        mae = float(np.mean(np.abs(preds - y_test)))

        # Cross-validation R²
        cv = cross_val_score(model, X_full, y_full, cv=5, scoring="r2")
        cv_mean = float(np.mean(cv))

        # Retrain on full data for production
        model.fit(X_full, y_full)

        self.models[key] = model
        self.scores[key] = round(r2, 4)
        self.cv_scores[key] = round(cv_mean, 4)
        self.mae_scores[key] = round(mae, 2)

        print(f"   ✅ {MODEL_INFO[key]['name']}: R²={r2:.4f}  CV-R²={cv_mean:.4f}  MAE={mae:.2f} yrs")

    # ─── Loading ───

    def _load_models(self):
        print("📂 Loading cached models...")
        cache = joblib.load(self.CACHE_PATH)
        self.models = cache["models"]
        self.scores = cache["scores"]
        self.cv_scores = cache["cv_scores"]
        self.mae_scores = cache["mae_scores"]
        self.scaler = joblib.load(self.SCALER_PATH)

        if os.path.exists(self.DATA_PATH):
            self.data = pd.read_csv(self.DATA_PATH)
        else:
            self.data = generate_synthetic_dataset()

        # Rebuild explainers
        X_scaled = self.scaler.transform(self.data[self.feature_names].values)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_scaled,
            feature_names=self.feature_names,
            class_names=["Brain Age"],
            mode="regression",
            random_state=42
        )
        
        for key, model in self.models.items():
            mtype = MODEL_INFO[key]["type"]
            if mtype == "tree":
                self.explainers[key] = shap.TreeExplainer(model)
            else:
                bg = shap.sample(pd.DataFrame(X_scaled, columns=self.feature_names), 50)
                self.explainers[key] = shap.KernelExplainer(model.predict, bg)

        self._print_summary()

    def _print_summary(self):
        print("\n" + "─" * 50)
        print("  MODEL PERFORMANCE SUMMARY")
        print("─" * 50)
        print(f"  {'Model':<25} {'R²':>8} {'CV-R²':>8} {'MAE':>8}")
        print("  " + "─" * 49)
        for key, info in MODEL_INFO.items():
            if key == "ensemble": continue
            print(f"  {info['name']:<25} {self.scores[key]:>8.4f} {self.cv_scores[key]:>8.4f} {self.mae_scores[key]:>6.2f}y")
        print("─" * 50 + "\n")

    # ─── Prediction ───

    def predict_with_explanation(self, features_dict, chronological_age=None, model_key="ensemble"):
        """
        Predict brain age with SHAP explanation using the selected model.
        
        model_key: one of 'random_forest', 'gradient_boosting', 'svr', 'lasso',
                   'ridge', 'elastic_net', 'knn', 'mlp', 'bayesian_ridge', 'ensemble'
        """
        feature_values = np.array([features_dict[f] for f in self.feature_names]).reshape(1, -1)
        feature_values_scaled = self.scaler.transform(feature_values)

        if model_key == "ensemble":
            res = self._predict_ensemble(feature_values, feature_values_scaled, chronological_age)
        else:
            res = self._predict_single(feature_values, feature_values_scaled, chronological_age, model_key)
            
        # ─── Attach LIME Explanation ───
        def predict_fn(X):
            if model_key == "ensemble":
                individual_keys = [k for k in MODEL_INFO.keys() if k != "ensemble"]
                wts = np.array([max(0.01, self.scores[k]) for k in individual_keys])
                wts = wts / wts.sum()
                all_preds = np.array([self.models[k].predict(X) for k in individual_keys])
                return np.sum(all_preds * wts[:, np.newaxis], axis=0)
            else:
                return self.models[model_key].predict(X)

        lime_exp = self.lime_explainer.explain_instance(
            data_row=feature_values_scaled[0],
            predict_fn=predict_fn,
            num_features=len(self.feature_names)
        )
        lime_weights = {self.feature_names[idx]: float(weight) for idx, weight in lime_exp.local_exp[1]}
        
        for c in res["feature_contributions"]:
            c["lime_value"] = lime_weights.get(c["feature"], 0.0)

        return res

    def _predict_single(self, feature_values, feature_values_scaled, chronological_age, model_key):
        """Prediction using a single model."""
        model = self.models[model_key]
        explainer = self.explainers[model_key]

        predicted_age = float(model.predict(feature_values_scaled)[0])

        # SHAP values
        sv = explainer.shap_values(feature_values_scaled)
        if isinstance(sv, list):
            shap_vals = sv[0]
        elif sv.ndim > 1:
            shap_vals = sv[0]
        else:
            shap_vals = sv

        # Base value
        ev = explainer.expected_value
        base_value = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)

        contributions = self._build_contributions(feature_values, shap_vals, base_value, chronological_age)

        brain_age_gap = round(predicted_age - chronological_age, 1) if chronological_age else None

        return {
            "predicted_age": round(predicted_age, 1),
            "chronological_age": chronological_age,
            "brain_age_gap": brain_age_gap,
            "base_value": round(base_value, 1),
            "feature_contributions": contributions,
            "model_key": model_key,
            "model_name": MODEL_INFO[model_key]["name"],
            "model_description": MODEL_INFO[model_key]["description"],
            "model_r2": self.scores[model_key],
            "model_cv_r2": self.cv_scores[model_key],
            "model_mae": self.mae_scores[model_key],
            "all_model_scores": self._get_all_scores(),
        }

    def _predict_ensemble(self, feature_values, feature_values_scaled, chronological_age):
        """Weighted average prediction across all models."""
        individual_keys = [k for k in MODEL_INFO.keys() if k != "ensemble"]

        # Weight by R² (higher R² = more weight)
        weights = np.array([max(0.01, self.scores[k]) for k in individual_keys])
        weights = weights / weights.sum()

        # Predictions from each model
        preds = {}
        for key in individual_keys:
            preds[key] = float(self.models[key].predict(feature_values_scaled)[0])

        # Weighted average
        predicted_age = sum(preds[k] * w for k, w in zip(individual_keys, weights))

        # Use the best tree model's SHAP for explanation
        best_tree = max(
            [k for k in individual_keys if MODEL_INFO[k]["type"] == "tree"],
            key=lambda k: self.scores[k]
        )
        explainer = self.explainers[best_tree]
        sv = explainer.shap_values(feature_values_scaled)
        shap_vals = sv[0] if (isinstance(sv, list) or sv.ndim > 1) else sv
        ev = explainer.expected_value
        base_value = float(ev[0]) if isinstance(ev, (list, np.ndarray)) else float(ev)

        contributions = self._build_contributions(feature_values, shap_vals, base_value, chronological_age)

        brain_age_gap = round(predicted_age - chronological_age, 1) if chronological_age else None

        # Ensemble "scores" = average of individual scores
        ens_r2 = round(float(np.mean([self.scores[k] for k in individual_keys])), 4)
        ens_cv = round(float(np.mean([self.cv_scores[k] for k in individual_keys])), 4)
        ens_mae = round(float(np.mean([self.mae_scores[k] for k in individual_keys])), 2)

        return {
            "predicted_age": round(predicted_age, 1),
            "chronological_age": chronological_age,
            "brain_age_gap": brain_age_gap,
            "base_value": round(base_value, 1),
            "feature_contributions": contributions,
            "model_key": "ensemble",
            "model_name": MODEL_INFO["ensemble"]["name"],
            "model_description": MODEL_INFO["ensemble"]["description"],
            "model_r2": ens_r2,
            "model_cv_r2": ens_cv,
            "model_mae": ens_mae,
            "individual_predictions": {k: round(preds[k], 1) for k in individual_keys},
            "ensemble_weights": {k: round(float(w), 3) for k, w in zip(individual_keys, weights)},
            "all_model_scores": self._get_all_scores(),
        }

    def _build_contributions(self, feature_values, shap_vals, base_value, chronological_age):
        contributions = []
        
        expected_deviation = 0.0
        if chronological_age is not None:
            expected_deviation = chronological_age - base_value
            
        # Distribute the expected deviation equally across all features
        expected_shap_per_feature = expected_deviation / len(self.feature_names)

        for i, fname in enumerate(self.feature_names):
            raw_shap = float(shap_vals[i])
            adj_shap = raw_shap - expected_shap_per_feature
            
            contributions.append({
                "feature": fname,
                "display_name": self.feature_display_names[fname],
                "value": float(feature_values[0, i]),
                "unit": FEATURE_UNITS[fname],
                "shap_value": raw_shap,
                "adjusted_shap_value": adj_shap,
                "direction": "aging" if raw_shap > 0 else "youthful",
            })
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions

    def _get_all_scores(self):
        """Return all model scores for the comparison table."""
        all_scores = {}
        for key in MODEL_INFO.keys():
            if key == "ensemble": continue
            all_scores[key] = {
                "name": MODEL_INFO[key]["name"],
                "short": MODEL_INFO[key]["short"],
                "r2": self.scores[key],
                "cv_r2": self.cv_scores[key],
                "mae": self.mae_scores[key],
            }
        return all_scores

    # ─── Sample Data ───

    def get_sample_subjects(self, n=15):
        if self.data is None:
            return []
        samples = []
        for age_bin in [(19, 30), (30, 40), (40, 50), (50, 58), (58, 66)]:
            subset = self.data[
                (self.data["chronological_age"] >= age_bin[0]) &
                (self.data["chronological_age"] < age_bin[1])
            ]
            for i in range(min(3, len(subset))):
                row = subset.iloc[i]
                samples.append({
                    "subject_id": row["subject_id"],
                    "gender": row["gender"],
                    "chronological_age": float(row["chronological_age"]),
                    "features": {f: float(row[f]) for f in self.feature_names},
                })
        return samples[:n]

    def get_available_models(self):
        """Return model metadata for the UI."""
        result = []
        for key in MODEL_INFO.keys():
            info = MODEL_INFO[key].copy()
            info["key"] = key
            if key != "ensemble":
                info["r2"] = self.scores[key]
                info["cv_r2"] = self.cv_scores[key]
                info["mae"] = self.mae_scores[key]
            result.append(info)
        return result


if __name__ == "__main__":
    model = BrainAgeModel()
    samples = model.get_sample_subjects()
    if samples:
        s = samples[2]  # middle-aged subject
        print(f"\nTest: {s['subject_id']} (age {s['chronological_age']})")
        for mkey in ["random_forest", "gradient_boosting", "svr", "lasso", "ensemble"]:
            result = model.predict_with_explanation(s["features"], s["chronological_age"], mkey)
            print(f"  {result['model_name']:>25}: predicted={result['predicted_age']}, gap={result['brain_age_gap']}")
