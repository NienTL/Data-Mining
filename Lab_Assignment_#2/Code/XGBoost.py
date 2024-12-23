import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt

# Read the data
train_X = pd.read_csv('./train_X.csv')
test_X = pd.read_csv('./test_X.csv')
train_y = pd.read_csv('./train_y.csv')

# Find numeric and categorical columns respectively
numeric_cols = train_X.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = train_X.select_dtypes(include=['object']).columns

# Fill the loss value of numeric data
num_imputer = SimpleImputer(strategy='mean') #mean 比 median 好
train_X[numeric_cols] = num_imputer.fit_transform(train_X[numeric_cols])
test_X[numeric_cols] = num_imputer.transform(test_X[numeric_cols])

# Fill the loss value of categorical data
cat_imputer = SimpleImputer(strategy='most_frequent')
train_X[categorical_cols] = cat_imputer.fit_transform(train_X[categorical_cols])
test_X[categorical_cols] = cat_imputer.transform(test_X[categorical_cols])

# One-Hot Encoding
train_X_encoded = pd.get_dummies(train_X, columns=categorical_cols, drop_first=True)
test_X_encoded = pd.get_dummies(test_X, columns=categorical_cols, drop_first=True)

# Align the features of train dataset and test dataset
# train_X_encoded, test_X_encoded = train_X_encoded.align(test_X_encoded, join='left', axis=1, fill_value=0)

train_y = train_y['has_died'] if 'has_died' in train_y.columns else train_y

scale_pos_weight = len(train_y[train_y == 0]) / len(train_y[train_y == 1])
# Initialize XGBoost model
xgb_model = XGBClassifier(
    objective='binary:logistic',
    random_state=42,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

param_dist = {
    'n_estimators': randint(1000, 3000),
    'max_depth': randint(10, 15),
    'learning_rate': uniform(0.01, 0.05),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'min_child_weight': randint(5, 15),
    'reg_alpha': uniform(0, 5),
    'reg_lambda': uniform(0, 5)
}

# Use random search to optimize parameters
random_search = RandomizedSearchCV(
    estimator=xgb_model, 
    param_distributions=param_dist, 
    n_iter=150, #from 50 to 75 to 150
    scoring='roc_auc', 
    cv=5, 
    random_state=42,    
    n_jobs=-1
)

random_search.fit(train_X_encoded, train_y)
best_xgb_model = random_search.best_estimator_

print("Best Parameters for XGBoost:", random_search.best_params_)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scorer = make_scorer(roc_auc_score, response_method='predict_proba')
f1_scorer = make_scorer(f1_score, average='macro')

roc_auc_scores = cross_val_score(best_xgb_model, train_X_encoded, train_y, cv=kf, scoring=roc_auc_scorer)
f1_scores = cross_val_score(best_xgb_model, train_X_encoded, train_y, cv=kf, scoring=f1_scorer)

print("Average ROC AUC:", np.mean(roc_auc_scores))
print("Average F1 Score:", np.mean(f1_scores))

# Train the model an do the prediction
best_xgb_model.fit(train_X_encoded, train_y)
test_predictions = best_xgb_model.predict(test_X_encoded)


submission = pd.DataFrame({
    'patient_id': test_X['patient_id'].astype(int),
    'has_died': test_predictions
})
submission.to_csv('./testing_result.csv', index=False)
print("Predictions saved to testing_result.csv")

# Find important features
feature_importances = best_xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': train_X_encoded.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# The top 20 important features
top_features = importance_df.head(20)
print("Top 20 Features by Importance:")
print(top_features)

plt.figure(figsize=(10, 8))
plt.barh(top_features['Feature'], top_features['Importance'], color="#FF7700")
plt.gca().invert_yaxis()
plt.title('Top 20 Most Important Features')
plt.xlabel('Importance Score')
plt.ylabel('Features')

for i, (feature, importance) in enumerate(zip(top_features['Feature'], top_features['Importance'])):
    plt.text(importance + 0.01, i, f'{importance:.4f}', va='center', fontsize=10, color='black')

plt.tight_layout()
plt.savefig('./top_features.png')
plt.show()