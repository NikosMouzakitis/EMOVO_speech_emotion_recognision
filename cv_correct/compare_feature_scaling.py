# plotting_script.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
        'Variance',
        'intensity_skewness',
        'intensity_outlier_score',
        'high_intensity_area',
        'max_circularity',
        'top3_circularity_mean',    
        'solidity_outlier',
        'abnormal_area_ratio',
        'circular_area_score',
        'asymmetry_score',
        'asymmetry_outlier',
        'boundary_sharpness_mean',
        'boundary_sharpness_max',
        'boundary_sharpness_outlier',
        'low_freq_energy',
#        'low_freq_entropy',
        'low_freq_mean',
        'low_freq_std',
        'low_freq_skewness',
        'mid_freq_energy',
#        'mid_freq_entropy',
        'mid_freq_mean',
        'mid_freq_std',
        'mid_freq_skewness',
        'high_freq_energy',
#        'high_freq_entropy',
        'high_freq_mean',
        'high_freq_std',
        'high_freq_skewness'
        ]


# Load the data
unnorm = pd.read_csv('features_unnormalized_train.csv')
norm = pd.read_csv('features_normalized_train.csv')

# Create output directory
os.makedirs('feature_distributions', exist_ok=True)

# Plot each feature
for feature in SELECTED_FEATURES:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(unnorm[feature], kde=True, color='royalblue')
    plt.title(f'Raw {feature[:30]}...' if len(feature) > 30 else f'Raw {feature}')
    
    plt.subplot(1, 2, 2)
    sns.histplot(norm[feature], kde=True, color='limegreen')
    plt.title(f'Normalized {feature[:30]}...' if len(feature) > 30 else f'Normalized {feature}')
    
    plt.tight_layout()
    plt.savefig(f'feature_distributions/{feature.replace("/", "_")}.png', dpi=120)
    plt.close()

print(f"Generated {len(SELECTED_FEATURES)} distribution plots")
