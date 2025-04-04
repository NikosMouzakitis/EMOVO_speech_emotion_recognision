import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("features/project_emovo_features_2000.csv")
emotion_encoder = LabelEncoder()
df["emotion_label"] = emotion_encoder.fit_transform(df["emotion"])
emotion_names = emotion_encoder.classes_

# Features and labels
features = ["DoC", "Density", "CC", "L", "M", "Avg_A"]
X = df[features]
y = df["emotion_label"]
groups = df["speaker"]

# Initialize SMOTE (we'll use it inside the loop)
smote = SMOTE(random_state=42)

# Random Forest Code 
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'class_weight': ['balanced', None],
    'max_features': ['sqrt', 'log2']
}

logo = LeaveOneGroupOut()
rf_results = {emotion: [] for emotion in emotion_names}

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    speaker_left_out = groups.iloc[test_idx[0]]
    
    # Apply SMOTE to training data only
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # RF training with resampled data
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, 
        param_grid_rf, 
        cv=5, 
        scoring='f1_macro',
        n_jobs=-1
    )
    grid_search.fit(X_train_res, y_train_res)  # Use resampled data
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store RF results
    for emotion_idx, emotion in enumerate(emotion_names):
        tp = cm[emotion_idx, emotion_idx]
        fp = cm[:, emotion_idx].sum() - tp
        fn = cm[emotion_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        rf_results[emotion].append({
            'Speaker': speaker_left_out,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'Best_Params': str(grid_search.best_params_)
        })

# =================================================================
# SVM Code
# =================================================================
param_dist_svm = {
    'kernel': ['linear', 'rbf'],  # Test for  both kernels
    'C': [0.1, 1, 10],           
    'gamma': ['scale', 0.01],     
    'class_weight': [None, 'balanced']
}

svm_results = {emotion: [] for emotion in emotion_names}

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    speaker_left_out = groups.iloc[test_idx[0]]
    
    # Apply SMOTE to training data only
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Feature scaling (critical for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)  # Scale resampled data
    X_test_scaled = scaler.transform(X_test)
    
    # SVM with updated parameter grid
    svm = SVC(random_state=42)
    svm_search = RandomizedSearchCV(
        svm,
        param_dist_svm,
        n_iter=10,  # Test 10 random combinations
        cv=3,       # Not many folds for speed
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42
    )
    svm_search.fit(X_train_scaled, y_train_res)  # Use resampled data
    best_svm = svm_search.best_estimator_
    y_pred = best_svm.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store SVM results
    for emotion_idx, emotion in enumerate(emotion_names):
        tp = cm[emotion_idx, emotion_idx]
        fp = cm[:, emotion_idx].sum() - tp
        fn = cm[emotion_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        svm_results[emotion].append({
            'Speaker': speaker_left_out,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1-Score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'Best_Params': str(svm_search.best_params_)
        })

# =================================================================
# Results Analysis
# =================================================================
def print_results(model_name, results_dict):
    print(f"\n=== {model_name} Results ===")
    for emotion in emotion_names:
        emotion_df = pd.DataFrame(results_dict[emotion])
        numeric_cols = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
        avg_results = emotion_df.groupby('Speaker')[numeric_cols].mean()
        
        print(f"\nEmotion: {emotion}")
        print(f"Best Parameters: {emotion_df['Best_Params'].mode()[0]}")
        print(avg_results)
        print("\nAverage Across Speakers:")
        print(avg_results.mean(axis=0))

print_results("Random Forest (with SMOTE optim)", rf_results)
print_results("SVM (with SMOTE optim)", svm_results)
