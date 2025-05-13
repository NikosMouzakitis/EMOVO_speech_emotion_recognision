import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from scipy import stats

def get_brain_mask(img):
    """Extract brain region while excluding skull"""
    # Adaptive thresholding to handle varying brightness
    thresh = cv2.adaptiveThreshold(img, 255, 
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 151, 7)
    
    # Morphological operations to clean up
    kernel = np.ones((15,15), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    
    # Keep largest connected component (brain)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    if n_labels < 2:
        return np.zeros_like(img)  # No brain found
    
    brain_mask = np.zeros_like(img)
    brain_mask[labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))] = 255
    
    # Fill holes and smooth edges
    return cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, np.ones((20,20), np.uint8))

def get_tumor_candidates(img, brain_mask, 
                        min_intensity_percentile=85,
                        min_size_frac=0.01, 
                        max_size_frac=0.15,
                        min_circularity=0.4):
    """Detect tumor candidates within brain mask with size/intensity constraints"""
    # Get intensity threshold (relative to brain tissue)
    brain_pixels = img[brain_mask > 0]
    if len(brain_pixels) == 0:
        return np.zeros_like(img)
    
    intensity_thresh = np.percentile(brain_pixels, min_intensity_percentile)
    
    # Create candidate mask
    candidates = np.zeros_like(img)
    candidates[(img > intensity_thresh) & (brain_mask > 0)] = 255
    
    # Size constraints
    brain_area = np.sum(brain_mask > 0)
    min_pixels = int(brain_area * min_size_frac)
    max_pixels = int(brain_area * max_size_frac)
    
    # Filter connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidates)
    tumor_mask = np.zeros_like(img)
    
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_pixels <= area <= max_pixels:
            # Check circularity
            contour, _ = cv2.findContours(
                (labels == i).astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contour:
                perimeter = cv2.arcLength(contour[0], True)
                circularity = (4 * np.pi * area) / (perimeter**2 + 1e-5)
                if circularity >= min_circularity:
                    tumor_mask[labels == i] = 255
    
    return tumor_mask

def extract_tumor_metrics(img_path, show_plots=False):
    """Complete tumor metrics extraction pipeline"""
    # Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Get masks
    brain_mask = get_brain_mask(img)
    tumor_mask = get_tumor_candidates(img, brain_mask)
    
    # Initialize metrics
    metrics = {
        'image': img_path.split('/')[-1],
        'has_tumor': np.any(tumor_mask)
    }
    
    # 1. Shape Metrics
    if metrics['has_tumor']:
        contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(contours, key=cv2.contourArea)
        
        metrics.update({
            'tumor_area': cv2.contourArea(cnt),
            'tumor_perimeter': cv2.arcLength(cnt, True),
            'tumor_circularity': (4 * np.pi * cv2.contourArea(cnt)) / (cv2.arcLength(cnt, True)**2 + 1e-5),
            'tumor_solidity': cv2.contourArea(cnt) / cv2.contourArea(cv2.convexHull(cnt))
        })
    
    # 2. Intensity Metrics
    brain_pixels = img[brain_mask > 0]
    metrics.update({
        'brain_mean_intensity': np.mean(brain_pixels),
        'brain_intensity_std': np.std(brain_pixels),
        'brain_skewness': stats.skew(brain_pixels.flatten())
    })
    
    if metrics['has_tumor']:
        tumor_pixels = img[tumor_mask > 0]
        metrics.update({
            'tumor_mean_intensity': np.mean(tumor_pixels),
            'tumor_intensity_std': np.std(tumor_pixels),
            'intensity_ratio': np.mean(tumor_pixels) / (np.mean(brain_pixels) + 1e-5),
            'relative_std': np.std(tumor_pixels) / (np.std(brain_pixels) + 1e-5)
        })
    
    # 3. Texture Metrics
    gray_img = np.uint8(img / 16)  # Reduce to 16 levels
    glcm = graycomatrix(gray_img, distances=[1, 3], angles=[0], levels=16)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        metrics[f'texture_{prop}'] = graycoprops(glcm, prop)[0, 0]
    
    # 4. Asymmetry Metrics
    left_hemi = img[:, :img.shape[1]//2]
    right_hemi = cv2.flip(img[:, img.shape[1]//2:], 1)
    metrics['asymmetry'] = np.mean(np.abs(left_hemi - right_hemi))
    
    # Visualization
    if show_plots:
        visualize_metrics(img, brain_mask, tumor_mask, metrics)
    
    return metrics

def visualize_metrics(img, brain_mask, tumor_mask, metrics):
    """Generate diagnostic plots"""
    plt.figure(figsize=(18,6))
    
    # Original with overlays
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.imshow(brain_mask, alpha=0.2, cmap='Blues')
    if np.any(tumor_mask):
        plt.imshow(tumor_mask, alpha=0.5, cmap='Reds')
    plt.title('Tumor Detection')
    
    # Intensity distribution
    plt.subplot(132)
    brain_pixels = img[brain_mask > 0]
    plt.hist(brain_pixels, bins=50, alpha=0.5, label='Brain')
    if np.any(tumor_mask):
        tumor_pixels = img[tumor_mask > 0]
        plt.hist(tumor_pixels, bins=50, alpha=0.5, label='Tumor')
    plt.legend()
    plt.title('Intensity Distribution')
    
    # Key metrics
    plt.subplot(133)
    plt.axis('off')
    metric_text = "\n".join([
        f"Tumor Area: {metrics.get('tumor_area',0):.0f} px",
        f"Circularity: {metrics.get('tumor_circularity',0):.2f}",
        f"Intensity Ratio: {metrics.get('intensity_ratio',0):.2f}",
        f"Asymmetry: {metrics.get('asymmetry',0):.1f}",
        f"Texture Contrast: {metrics.get('texture_contrast',0):.1f}"
    ])
    plt.text(0.1, 0.5, metric_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{metrics['image']}_metrics.png", dpi=300)
    plt.close()

# Example usage
if __name__ == "__main__":
    # Analyze tumor case
    tumor_metrics = extract_tumor_metrics("thismeningioma/Tr-me_0138.jpg", show_plots=True)
    
    # Analyze normal case
    normal_metrics = extract_tumor_metrics("thisnotumor/Tr-no_0054.jpg", show_plots=True)
    
    # Print comparison
    print("\nTumor Case Metrics:")
    print({k:v for k,v in tumor_metrics.items() if not k.startswith('texture_')})
    
    print("\nNormal Case Metrics:")
    print({k:v for k,v in normal_metrics.items() if not k.startswith('texture_')})
