from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import numpy as np

class MLModels:
    def __init__(self):
        # Initialize models with optimized hyperparameters
        self.logistic = LogisticRegression(
            max_iter=1000,
            C=0.1,
            class_weight='balanced'
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        self.gradient_boost = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        self.xgboost = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            scale_pos_weight=1,
            random_state=42
        )
        self.models = {}
        self.feature_importances = None
        self.cross_val_scores = {}

    def train_models(self, X, y):
        """Train models with cross-validation and ensemble voting"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train and evaluate each model
        models_list = {
            'Logistic Regression': self.logistic,
            'Random Forest': self.random_forest,
            'Gradient Boosting': self.gradient_boost,
            'XGBoost': self.xgboost
        }

        for name, model in models_list.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            self.cross_val_scores[name] = cv_scores.mean()

            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_score': cv_scores.mean(),
                'report': classification_report(y_test, y_pred)
            }

        # Store feature importance from Random Forest
        self.feature_importances = self.random_forest.feature_importances_

        return self.models

    def predict_match(self, X_match):
        """Predict match outcome using ensemble voting"""
        predictions = {}
        ensemble_probs = np.zeros(2)

        for name, model_info in self.models.items():
            model = model_info['model']
            prob = model.predict_proba(X_match)[0]
            predictions[name] = {
                'win_probability': prob[1],
                'confidence': max(prob)
            }
            # Weight predictions by model accuracy
            ensemble_probs += prob * model_info['cv_score']

        # Normalize ensemble probabilities
        ensemble_probs /= sum(self.cross_val_scores.values())

        # Add ensemble prediction
        predictions['Ensemble'] = {
            'win_probability': ensemble_probs[1],
            'confidence': max(ensemble_probs)
        }

        return predictions

    def get_feature_importance(self):
        """Get feature importance from Random Forest model"""
        return self.feature_importances