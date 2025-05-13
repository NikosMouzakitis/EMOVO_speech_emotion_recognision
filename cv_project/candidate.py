import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

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


def analyze_mri_with_size_constraints(img_path, min_tumor_frac=0.015, max_tumor_frac=0.13):
    # Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Get brain mask (same as before)
    brain_mask = extract_brain_mask(img)
    brain_area = np.sum(brain_mask > 0)
    
    # Calculate absolute pixel limits
    min_tumor_pixels = int(brain_area * min_tumor_frac)
    max_tumor_pixels = int(brain_area * max_tumor_frac)
    
    # Find tumor candidates with size constraints
    candidates = find_tumor_candidates(img, brain_mask, min_tumor_pixels, max_tumor_pixels)
    
    # Feature extraction
    features = {
        'brain_area': brain_area,
        'tumor_candidates': len(candidates),
        **extract_tumor_features(img, candidates)
    }
    
    # Visualization
    visualize_constrained_results(img, brain_mask, candidates, features)
    return features

def find_tumor_candidates(img, brain_mask, min_pixels, max_pixels):
    """Find tumor regions with strict size constraints"""
    # Enhanced tumor detection
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.bitwise_and(thresh, brain_mask)
    
    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Connected components with size filtering
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    candidates = []
    
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_pixels <= area <= max_pixels:
            mask = (labels == i).astype(np.uint8)
            
            # Additional circularity check
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = contours[0]
                perimeter = cv2.arcLength(cnt, True)
                circularity = (4 * np.pi * area) / (perimeter**2 + 1e-5)
                
                if circularity > 0.5:  # Only keep roundish regions
                    candidates.append({
                        'mask': mask,
                        'area': area,
                        'circularity': circularity,
                        'centroid': (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP])
                    })
    
    return candidates

def extract_tumor_features(img, candidates):
    """Extract features from valid tumor candidates"""
    if not candidates:
        return {
            'tumor_area': 0,
            'tumor_circularity': 0,
            'tumor_intensity': 0
        }
    
    # Get primary tumor candidate (largest valid region)
    main_tumor = max(candidates, key=lambda x: x['area'])
    tumor_pixels = img[main_tumor['mask'] > 0]
    
    return {
        'tumor_area': main_tumor['area'],
        'tumor_circularity': main_tumor['circularity'],
        'tumor_intensity': np.mean(tumor_pixels),
        'tumor_intensity_std': np.std(tumor_pixels)
    }

def visualize_constrained_results(img, brain_mask, candidates, features):
    """Visualization with size constraints highlighted"""
    plt.figure(figsize=(15,5))
    
    # Original with brain contour
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    brain_contour = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in brain_contour:
        plt.plot(cnt[:,0,0], cnt[:,0,1], 'g-', linewidth=1)
    plt.title('Original MRI')
    
    # Tumor candidates
    plt.subplot(132)
    plt.imshow(img, cmap='gray')
    for cand in candidates:
        contour = cv2.findContours(cand['mask'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in contour:
            plt.plot(cnt[:,0,0], cnt[:,0,1], 'r-', linewidth=2)
    plt.title(f'{len(candidates)} Tumor Candidates')
    
    # Feature summary
    plt.subplot(133)
    plt.axis('off')
    text = f"""
    Brain Area: {features['brain_area']} px
    Tumor Area: {features['tumor_area']} px
    Circularity: {features.get('tumor_circularity',0):.2f}
    Intensity: {features.get('tumor_intensity',0):.1f}±{features.get('tumor_intensity_std',0):.1f}
    """
    plt.text(0.1, 0.5, text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('constrained_tumor_detection.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
#features = analyze_mri_with_size_constraints("thisnotumor/Tr-no_0028.jpg")
features = analyze_mri_with_size_constraints("thismeningioma/Tr-me_0016.jpg")

print("Detection complete. Features extracted:")
for k, v in features.items():
    print(f"{k:20}: {v}")
