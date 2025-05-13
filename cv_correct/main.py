import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import cv2
from radiomics import featureextractor, firstorder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score)
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc



# Configuration
SELECTED_FEATURES = [
        'original_firstorder_10Percentile', 
        'original_firstorder_90Percentile', 
        'original_firstorder_Energy', 
        'original_firstorder_Entropy', 
        'original_firstorder_InterquartileRange', 
        'original_firstorder_Kurtosis', 
        'original_firstorder_Maximum',
        'original_firstorder_MeanAbsoluteDeviation',
        'original_firstorder_Mean',
        'original_firstorder_Median',
        'original_firstorder_Minimum',
        'original_firstorder_Range',
        'original_firstorder_RobustMeanAbsoluteDeviation',
        'original_firstorder_RootMeanSquared',
        'original_firstorder_Skewness',
        'original_firstorder_TotalEnergy',
        'original_firstorder_Uniformity',
        'original_firstorder_Variance',
        'original_glcm_Autocorrelation',
        'original_glcm_ClusterProminence',
        'original_glcm_ClusterShade',
        'original_glcm_ClusterTendency',
        'original_glcm_Contrast',
        'original_glcm_Correlation',
        'original_glcm_DifferenceAverage',
        'original_glcm_DifferenceEntropy',
        'original_glcm_DifferenceVariance',
        'original_glcm_Id',
        'original_glcm_Idm',
        'original_glcm_Idmn',
        'original_glcm_Idn',
        'original_glcm_Imc1',
        'original_glcm_Imc2',
        'original_glcm_InverseVariance',
        'original_glcm_JointAverage',
        'original_glcm_JointEnergy',
        'original_glcm_JointEntropy',
        'original_glcm_MCC',
        'original_glcm_MaximumProbability',
        'original_glcm_SumAverage',
        'original_glcm_SumEntropy',
        'original_glcm_SumSquares',
        'original_gldm_DependenceEntropy',
        'original_gldm_DependenceNonUniformity',
        'original_gldm_DependenceNonUniformityNormalized',
        'original_gldm_DependenceVariance',
        'original_gldm_GrayLevelNonUniformity',
        'original_gldm_GrayLevelVariance',
        'original_gldm_HighGrayLevelEmphasis',
        'original_gldm_LargeDependenceEmphasis',
        'original_gldm_LargeDependenceHighGrayLevelEmphasis',
        'original_gldm_LargeDependenceLowGrayLevelEmphasis',
        'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis',
        'original_gldm_SmallDependenceHighGrayLevelEmphasis',
        'original_gldm_SmallDependenceLowGrayLevelEmphasis',
        'original_glrlm_GrayLevelNonUniformity',
        'original_glrlm_GrayLevelNonUniformityNormalized',
        'original_glrlm_GrayLevelVariance',
        'original_glrlm_HighGrayLevelRunEmphasis',
        'original_glrlm_LongRunEmphasis',
        'original_glrlm_LongRunHighGrayLevelEmphasis',
        'original_glrlm_LongRunLowGrayLevelEmphasis',
        'original_glrlm_LowGrayLevelRunEmphasis',
        'original_glrlm_RunEntropy',
        'original_glrlm_RunLengthNonUniformity',
        'original_glrlm_RunLengthNonUniformityNormalized',
        'original_glrlm_RunPercentage',
        'original_glrlm_RunVariance',
        'original_glrlm_ShortRunEmphasis',
        'original_glrlm_ShortRunHighGrayLevelEmphasis',
        'original_glrlm_ShortRunLowGrayLevelEmphasis',
        'original_glszm_GrayLevelNonUniformity',
        'original_glszm_GrayLevelNonUniformityNormalized',
        'original_glszm_GrayLevelVariance',
        'original_glszm_HighGrayLevelZoneEmphasis',
        'original_glszm_LargeAreaEmphasis',
        'original_glszm_LargeAreaHighGrayLevelEmphasis',
        'original_glszm_LargeAreaLowGrayLevelEmphasis',
        'original_glszm_LowGrayLevelZoneEmphasis',
        'original_glszm_SizeZoneNonUniformity',
        'original_glszm_SizeZoneNonUniformityNormalized',
        'original_glszm_SmallAreaEmphasis',
        'original_glszm_SmallAreaHighGrayLevelEmphasis',
        'original_glszm_SmallAreaLowGrayLevelEmphasis',
        'original_glszm_ZoneEntropy',
        'original_glszm_ZonePercentage',
        'original_glszm_ZoneVariance',
        'original_ngtdm_Busyness',
        'original_ngtdm_Coarseness',
        'original_ngtdm_Complexity',
        'original_ngtdm_Contrast',
        'original_ngtdm_Strength',
        '10Percentile', 
        '90Percentile',
        'Energy',
        'Entropy',
        'InterquartileRange',
        'Kurtosis',
        'Maximum',
        'MeanAbsoluteDeviation',
        'Mean',
        'Median',
        'Minimum',
        'Range',
        'RobustMeanAbsoluteDeviation',
        'RootMeanSquared',
        'Skewness',
        'TotalEnergy',
        'Uniformity',
        'Variance'
#        'intensity_skewness',
#        'intensity_outlier_score',
#        'high_intensity_area',
#        'max_circularity',
#        'top3_circularity_mean',    
#        'solidity_outlier',
#        'abnormal_area_ratio',
#        'circular_area_score',
#        'asymmetry_score',
#        'asymmetry_outlier',
#        'boundary_sharpness_mean',
#        'boundary_sharpness_max',
#        'boundary_sharpness_outlier',
#        'low_freq_energy',
#        'low_freq_entropy',
#        'low_freq_mean',
#        'low_freq_std',
#        'low_freq_skewness',
#        'mid_freq_energy',
#        'mid_freq_entropy',
#        'mid_freq_mean',
#        'mid_freq_std',
#        'mid_freq_skewness',
#        'high_freq_energy',
#        'high_freq_entropy',
#        'high_freq_mean',
#        'high_freq_std',
#        'high_freq_skewness'
       ]

# ======================
# 1. Feature Extraction 
# (Your provided code)
# ======================
def extract_frequency_features(image_array):
    """Improved version with size-adaptive frequency bands"""
    if len(image_array.shape) > 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    rows, cols = image_array.shape
    max_radius = min(rows, cols) // 2  # Maximum meaningful frequency
    
    # Define bands as fractions of maximum frequency
    low_band = create_bandpass_mask(rows, cols, rows//2, cols//2, 0, 0.2*max_radius)
    mid_band = create_bandpass_mask(rows, cols, rows//2, cols//2, 0.2*max_radius, 0.6*max_radius)
    high_band = create_bandpass_mask(rows, cols, rows//2, cols//2, 0.6*max_radius, max_radius)
    
    fft = np.fft.fft2(image_array)
    fft_shifted = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shifted)
    
    rows, cols = image_array.shape
    crow, ccol = rows // 2, cols // 2
        
    features = {}
    features.update(get_band_features(magnitude_spectrum, low_band, 'low_freq'))
    features.update(get_band_features(magnitude_spectrum, mid_band, 'mid_freq'))
    features.update(get_band_features(magnitude_spectrum, high_band, 'high_freq'))
    
    return features

def create_bandpass_mask(rows, cols, crow, ccol, min_radius, max_radius):
    y, x = np.ogrid[:rows, :cols]
    mask = np.sqrt((x - ccol)**2 + (y - crow)**2)
    band_mask = np.logical_and(mask >= min_radius, mask <= max_radius)
    return band_mask

def get_band_features(spectrum, mask, prefix):
    band_spectrum = spectrum * mask
    band_spectrum_normalized = band_spectrum / (np.sum(band_spectrum) + 1e-10)
    
    energy = np.sum(band_spectrum**2)
    entropy = -np.sum(band_spectrum_normalized * np.log(band_spectrum_normalized + 1e-10))
    mean = np.mean(band_spectrum)
    std = np.std(band_spectrum)
    skewness = np.mean((band_spectrum - mean)**3) / (std**3 + 1e-10)
    
    return {
        f'{prefix}_energy': energy,
        f'{prefix}_entropy': entropy,
        f'{prefix}_mean': mean,
        f'{prefix}_std': std,
        f'{prefix}_skewness': skewness
    }

def enhanced_tumor_features(image_array):
    norm_img = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    blurred = cv2.medianBlur(norm_img, 5)
    
    features = {}
    features.update(intensity_analysis(norm_img))
    features.update(enhanced_cluster_detection(blurred))
    features.update(symmetry_analysis(norm_img))
    features.update(boundary_analysis(blurred))
    
    return features

def intensity_analysis(img):
    hist = cv2.calcHist([img], [0], None, [256], [0,256])
    hist = hist.flatten()/hist.sum()
    
    return {
        'intensity_skewness': (hist * np.arange(256)).mean(),
        'intensity_outlier_score': np.percentile(img, 99) - np.percentile(img, 90),
        'high_intensity_area': np.sum(img > np.percentile(img, 95))/img.size
    }

def enhanced_cluster_detection(img):
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circularities = []
    solidities = []
    areas = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50: continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
            
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area if hull_area > 0 else 0
        
        circularities.append(circularity)
        solidities.append(solidity)
        areas.append(area)
    
    return {
        'max_circularity': np.max(circularities) if circularities else 0,
        'top3_circularity_mean': np.mean(sorted(circularities)[-3:]) if circularities else 0,
        'solidity_outlier': np.percentile(solidities, 95) if solidities else 0,
        'abnormal_area_ratio': np.sum(areas)/(img.shape[0]*img.shape[1]) if areas else 0,
        'circular_area_score': np.sum([a*c for a,c in zip(areas, circularities)]) if areas else 0
    }

def symmetry_analysis(img):
    h, w = img.shape
    left = img[:, :w//2]
    right = img[:, w//2:]
    right_flipped = cv2.flip(right, 1)
    
    min_width = min(left.shape[1], right_flipped.shape[1])
    left = left[:, :min_width]
    right_flipped = right_flipped[:, :min_width]
    
    diff = cv2.absdiff(left, right_flipped)
    
    return {
        'asymmetry_score': np.mean(diff),
        'asymmetry_outlier': np.percentile(diff, 95)
    }

def boundary_analysis(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    return {
        'boundary_sharpness_mean': np.mean(magnitude),
        'boundary_sharpness_max': np.max(magnitude),
        'boundary_sharpness_outlier': np.percentile(magnitude, 95)
    }

def process_image(img_path):
    """Process a single image and return features"""
    array = sitk.GetArrayFromImage(sitk.ReadImage(img_path, sitk.sitkFloat32))
    if array.ndim == 3:
        array = np.mean(array, axis=0)
    image = sitk.GetImageFromArray(array)
    
    # Mask creation
    mask_array = np.ones_like(array, dtype=np.uint8)
    border_size = 10 
    mask_array[:border_size, :] = 0
    mask_array[-border_size:, :] = 0
    mask_array[:, :border_size] = 0
    mask_array[:, -border_size:] = 0
    mask = sitk.GetImageFromArray(mask_array)
    mask.CopyInformation(image)

    # Feature extraction
    extractor = featureextractor.RadiomicsFeatureExtractor()
    features = extractor.execute(image, mask)
    first_order_features = firstorder.RadiomicsFirstOrder(image, mask).execute()
    
    # Custom features
    processing_array = cv2.normalize(array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    custom_features = enhanced_tumor_features(processing_array)
    freq_features = extract_frequency_features(processing_array)
    
    return {
        **features,
        **first_order_features,
        **custom_features,
        **freq_features
    }

# ======================
# 2. Data Preparation
# ======================
def load_data(healthy_dir, tumor_dir):
    """Load and label all images from both directories"""
    healthy_files = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) if f.endswith('.jpg')]
    tumor_files = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if f.endswith('.jpg')]
    
    # Create DataFrame with labels (0=healthy, 1=tumor)
    healthy_df = pd.DataFrame({'filepath': healthy_files, 'label': 0})
    tumor_df = pd.DataFrame({'filepath': tumor_files, 'label': 1})
    return pd.concat([healthy_df, tumor_df], ignore_index=True)

# ======================
# 3. Main Pipeline
# ======================
def main():
    # 1. Load data
    data_df = load_data("../thismeningioma", "../thisglioma")
    #data_df = load_data("thisnotumor", "thismeningioma")
    #data_df = load_data("../thisnotumor", "../thismeningioma")
    
    # 2. Extract features (this will take time)
    print("Extracting features...")
    features = []
    for idx, row in data_df.iterrows():
        features.append(process_image(row['filepath']))
        if idx % 10 == 0:
            print(f"Processed {idx+1}/{len(data_df)} images")
    
    features_df = pd.DataFrame(features)
    
    # 3. Prepare features and labels
    X = features_df[SELECTED_FEATURES]
    y = data_df['label'].values
    
    # 4. Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 5. Normalize features (0-1 range)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
   
    # 5.1 Save unnormalized features so we can plot afterwards.
#    X_train.to_csv('features_unnormalized_train.csv', index=False)
#    X_test.to_csv('features_unnormalized_test.csv', index=False)
    # 5.2 Save normalized features so we can plot them as asked afterwards.
#    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('features_normalized_train.csv', index=False)
#    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('features_normalized_test.csv', index=False)

    # ======================
    # 6. Model Training
    # ======================
    
    # A) Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_rf = rf.predict(X_test_scaled)
    print("\nRandom Forest Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_rf):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    
    # B) Neural Network
    print("\nTraining Neural Network...")
    nn = MLPClassifier(hidden_layer_sizes=(100, 50), 
                      max_iter=500, 
                      random_state=42)
    nn.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred_nn = nn.predict(X_test_scaled)
    print("\nNeural Network Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.3f}")
    print(f"AUC: {roc_auc_score(y_test, y_pred_nn):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_nn))
    
    # ======================
    # 7. Visualization
    # ======================
    plt.figure(figsize=(12, 5))
    
    # Confusion Matrix for RF
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Meningioma', 'Glioma'],
                yticklabels=['Meningioma', 'Glioma'])
    plt.title('Random Forest Confusion Matrix')
    
    # Confusion Matrix for NN
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, y_pred_nn), 
                annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Meningioma', 'Glioma'],
                yticklabels=['Meningioma', 'Glioma'])
    plt.title('Neural Network Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # Feature Importance (Random Forest)
    plt.figure(figsize=(10, 8))
    importances = rf.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    plt.title('Top 20 Important Features (Random Forest)')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()



    # ======================
    # 8. ROC Curve Plotting
    # ======================
    # For ROC, we need predicted probabilities
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    y_prob_nn = nn.predict_proba(X_test_scaled)[:, 1]

    # Compute ROC curves
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)

    # Compute AUCs
    auc_rf = auc(fpr_rf, tpr_rf)
    auc_nn = auc(fpr_nn, tpr_nn)

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='blue')
    plt.plot(fpr_nn, tpr_nn, label=f'Neural Net (AUC = {auc_nn:.2f})', color='orange')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    main()
