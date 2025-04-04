import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline

def safe_divide(numerator, denominator):
    return numerator / denominator if denominator != 0 else 0

def evaluate_model(model, X, y, groups):
    logo = LeaveOneGroupOut()
    
    # Initialize accumulators
    male_true, male_pred = [], []
    female_true, female_pred = [], []
    
    print("\n" + "="*60)
    print(f"Starting evaluation for {model.__class__.__name__}")
    print("="*60)
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        left_out_speaker = groups[test_idx[0]]
        left_out_gender = "male" if y_test[0] == 1 else "female"
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Store predictions by gender
        if left_out_gender == "male":
            male_true.extend(y_test)
            male_pred.extend(y_pred)
        else:
            female_true.extend(y_test)
            female_pred.extend(y_pred)
        
        # Debug print for each fold
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_counts = dict(zip(unique, counts))
        male_pred_count = pred_counts.get(1, 0)
        female_pred_count = pred_counts.get(2, 0)
        
        print(f"\nFold {fold+1}: Left out {left_out_speaker} ({left_out_gender})")
        print(f"True males: {sum(y_test == 1)}, Females: {sum(y_test == 2)}")
        print(f"Predicted males: {male_pred_count}, Females: {female_pred_count}")
    
    # Convert to arrays
    male_true, male_pred = np.array(male_true), np.array(male_pred)
    female_true, female_pred = np.array(female_true), np.array(female_pred)
    
    # Print prediction distribution
    print("\n" + "-"*40)
    print("Male Test Samples:")
    male_conf = confusion_matrix(male_true, male_pred, labels=[1, 2])
    print(f"Confusion Matrix:\n{male_conf}")
    print(f"Male accuracy: {accuracy_score(male_true, male_pred):.3f}")
    
    print("\nFemale Test Samples:")
    female_conf = confusion_matrix(female_true, female_pred, labels=[1, 2])
    print(f"Confusion Matrix:\n{female_conf}")
    print(f"Female accuracy: {accuracy_score(female_true, female_pred):.3f}")
    print("-"*40 + "\n")
    
    # Calculate metrics
    male_metrics = {
        'Accuracy': accuracy_score(male_true, male_pred),
        'Sensitivity': safe_divide(male_conf[0, 0], male_conf[0, 0] + male_conf[0, 1]),  # TP / (TP + FN)
        'Specificity': safe_divide(female_conf[1, 1], female_conf[1, 0] + female_conf[1, 1]),  # TN / (TN + FP)
        'F1-Score': f1_score(male_true, male_pred, pos_label=1)
    }
    
    female_metrics = {
        'Accuracy': accuracy_score(female_true, female_pred),
        'Sensitivity': safe_divide(female_conf[1, 1], female_conf[1, 0] + female_conf[1, 1]),  # TP / (TP + FN)
        'Specificity': safe_divide(male_conf[0, 0], male_conf[0, 0] + male_conf[0, 1]),  # TN / (TN + FP)
        'F1-Score': f1_score(female_true, female_pred, pos_label=2)
    }
    
    return {'male': male_metrics, 'female': female_metrics}

# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_csv("features/project_emovo_features_3000.csv")
    X = df[['DoC', 'Density', 'CC', 'L', 'M', 'Avg_A']].values
    y = np.where(df['speaker'].str.startswith('m'), 1, 2)
    groups = df['speaker'].values
    
    pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        MinMaxScaler()
    )
    X_processed = pipe.fit_transform(X, y)

    # Initialize and evaluate models
    models = {
        'SVM': SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced'),
        'RF': RandomForestClassifier(n_estimators=200, max_depth=5,
                                   class_weight='balanced', random_state=42)
    }

    results = {}
    for name, model in models.items():
        print(f"\n{'#'*20} Evaluating {name} {'#'*20}")
        results[name] = evaluate_model(model, X_processed, y, groups)

    # Prepare final output
    output_data = []
    for gender in ['male', 'female']:
        for model in ['SVM', 'RF']:
            metrics = results[model][gender]
            output_data.append({
                'Gender': gender.capitalize(),
                'Classifier': model,
                'Accuracy (%)': metrics['Accuracy'] * 100,
                'Sensitivity': metrics['Sensitivity'],
                'Specificity': metrics['Specificity'],
                'F1-Score': metrics['F1-Score']
            })

    output_df = pd.DataFrame(output_data).round(3)
    output_df.to_csv("final_gender_results.csv", index=False)
    print("\nFinal Results:")
    print(output_df.to_markdown(index=False))

