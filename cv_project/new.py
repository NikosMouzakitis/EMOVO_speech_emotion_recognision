import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
import scipy.stats
from scipy.spatial.distance import jensenshannon

def analyze_mri(img_path):
    # 1. Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")
    
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Skull stripping
    brain_mask = extract_brain_mask(img)
    if np.sum(brain_mask) == 0:
        return None
    
    # 3. Extract brain ROI
    brain_roi = cv2.bitwise_and(img, img, mask=brain_mask)
    
    # 4. Calculate features
    features = {
        **get_shape_features(brain_mask),
        **get_intensity_features(brain_roi),
        **get_texture_features(brain_roi),
        **get_asymmetry_features(brain_roi, brain_mask),
        **get_tumor_candidate_features(brain_roi)
    }
    
    # 5. Visualization
    visualize_results(img, brain_mask, features)
    
    return features

def extract_brain_mask(img):
    """Improved skull stripping with morphological cleanup"""
    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(img, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 151, 7)
    
    # Morphological operations
    kernel = np.ones((15,15), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # Keep largest component
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if n_labels < 2:
        return np.zeros_like(img)
    
    brain_mask = np.zeros_like(img)
    brain_mask[labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = 255
    
    # Fill holes
    return cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))

def get_shape_features(mask):
    """Calculate brain shape characteristics"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    
    return {
        'brain_area': area,
        'brain_perimeter': perimeter,
        'brain_circularity': (4 * np.pi * area) / (perimeter**2 + 1e-5),
        'brain_convexity': cv2.contourArea(hull) / (area + 1e-5)
    }

def get_intensity_features(roi):
    """Analyze intensity distribution within brain"""
    pixels = roi[roi > 0]
    if len(pixels) == 0:
        return {}
    
    return {
        'intensity_mean': np.mean(pixels),
        'intensity_std': np.std(pixels),
        'intensity_skewness': scipy.stats.skew(pixels),
        'intensity_kurtosis': scipy.stats.kurtosis(pixels),
        'high_intensity_pixels': np.sum(pixels > np.percentile(pixels, 90))
    }

def get_texture_features(roi):
    """Calculate texture features using GLCM"""
    # Quantize image to 16 levels for texture analysis
    quantized = np.uint8(roi / 16)
    glcm = graycomatrix(quantized, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=16)
    
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = {}
    for prop in props:
        val = graycoprops(glcm, prop)
        features.update({
            f'texture_{prop}_dist1': val[0].mean(),
            f'texture_{prop}_dist3': val[1].mean()
        })
    
    return features

def get_asymmetry_features(roi, mask):
    """Quantify left-right asymmetry"""
    h, w = roi.shape
    left = roi[:, :w//2]
    right = cv2.flip(roi[:, w//2:], 1)
    
    # Mask alignment
    left_mask = mask[:, :w//2]
    right_mask = cv2.flip(mask[:, w//2:], 1)
    valid_mask = left_mask & right_mask
    
    if np.sum(valid_mask) == 0:
        return {'asymmetry': 1.0}  # Maximum asymmetry
    
    left_valid = left[valid_mask > 0]
    right_valid = right[valid_mask > 0]
    
    # Intensity distribution comparison
    hist_left = np.histogram(left_valid, bins=32, range=(0,255))[0]
    hist_right = np.histogram(right_valid, bins=32, range=(0,255))[0]
    
    return {
        'asymmetry_intensity': np.mean(np.abs(left_valid - right_valid)),
        'asymmetry_jsd': jensenshannon(hist_left, hist_right),
        'asymmetry_correlation': np.corrcoef(left_valid, right_valid)[0,1]
    }

def get_tumor_candidate_features(roi):
    """Detect and characterize suspicious regions"""
    # Adaptive thresholding for tumor candidates
    candidates = cv2.adaptiveThreshold(roi, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 7)
    
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    if n_labels < 2:
        return {'tumor_candidates': 0}
    
    features = {'tumor_candidates': n_labels - 1}
    for i in range(1, min(n_labels, 4)):  # Analyze top 3 candidates
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 10:
            continue
            
        component = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        
        features.update({
            f'tumor_{i}_area': area,
            f'tumor_{i}_circularity': (4 * np.pi * area) / (cv2.arcLength(cnt, True)**2 + 1e-5),
            f'tumor_{i}_intensity': np.mean(roi[component > 0])
        })
    
    return features

def visualize_results(img, mask, features):
    """Generate comprehensive visualization"""
    plt.figure(figsize=(18, 12))
    
    # Original and brain mask
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original MRI')
    
    plt.subplot(232)
    plt.imshow(mask, cmap='gray')
    plt.title('Brain Extraction')
    
    # Asymmetry visualization
    h, w = img.shape
    left = img[:, :w//2]
    right = cv2.flip(img[:, w//2:], 1)
    
    plt.subplot(233)
    plt.imshow(np.abs(left - right), cmap='hot')
    plt.title(f'Asymmetry Map (JSD={features.get("asymmetry_jsd",0):.3f})')
    
    # Tumor candidates
    candidates = cv2.bitwise_and(img, img, mask=(
        (img > np.percentile(img[mask>0], 90)).astype(np.uint8) * mask))
    
    plt.subplot(234)
    plt.imshow(img, cmap='gray')
    plt.imshow(candidates, alpha=0.5, cmap='hot')
    plt.title(f'{features.get("tumor_candidates",0)} Tumor Candidates')
    
    # Texture visualization
    glcm = graycomatrix(np.uint8(img/16), [3], [0], 16)
    texture = graycoprops(glcm, 'contrast')[0,0]
    
    plt.subplot(235)
    plt.imshow(cv2.Laplacian(img, cv2.CV_64F), cmap='jet')
    plt.title(f'Texture (Contrast={texture:.1f})')
    
    # Feature summary
    plt.subplot(236)
    plt.axis('off')
    feat_text = "\n".join([
        f"Circularity: {features.get('brain_circularity',0):.3f}",
        f"Asymmetry JSD: {features.get('asymmetry_jsd',0):.3f}",
        f"Candidates: {features.get('tumor_candidates',0)}",
        f"High Intensity: {features.get('high_intensity_pixels',0)}px",
        f"Texture Energy: {features.get('texture_energy_dist1',0):.3f}"
    ])
    plt.text(0, 0.5, feat_text, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('mri_analysis_report.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Example usage
if __name__ == "__main__":
    try:
        #features = analyze_mri("thisnotumor/Tr-no_0028.jpg")
        features = analyze_mri("thismeningioma/Tr-me_0157.jpg")
        if features:
            print("\nExtracted Features:")
            for k, v in features.items():
                print(f"{k:>25}: {v:.3f}" if isinstance(v, float) else f"{k:>25}: {v}")
            
            print("\nVisual report saved to 'mri_analysis_report.png'")
    except Exception as e:
        print(f"Error: {str(e)}")
