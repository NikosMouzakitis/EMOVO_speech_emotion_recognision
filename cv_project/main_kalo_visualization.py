import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_tumors(img_path, 
                 min_tumor_frac=0.01,  # 1% of brain area (adjust down if missing small tumors)
                 max_tumor_frac=0.15,   # 15% of brain area (adjust up if missing large tumors)
                 intensity_thresh=0.85, # % of brain intensity (0.8-0.95)
                 circularity_thresh=0.4): # 0=any shape, 1=perfect circle (0.3-0.7)
    
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # 1. Brain Extraction (Debuggable Steps)
    _, brain_mask = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15,15), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
    brain_area = np.sum(brain_mask > 0)
    print(f"Brain area: {brain_area}px")

    # 2. Intensity Threshold (Relative to Brain Tissue)
    brain_pixels = img[brain_mask > 0]
    if len(brain_pixels) == 0:
        return None
    
    tumor_intensity_thresh = np.percentile(brain_pixels, intensity_thresh * 100)
    print(f"Intensity threshold: {tumor_intensity_thresh:.1f} (Brain median: {np.median(brain_pixels):.1f})")
    
    # 3. Candidate Detection
    candidates = cv2.bitwise_and(img, img, mask=((img > tumor_intensity_thresh) & brain_mask).astype(np.uint8))
    _, tumor_candidates = cv2.threshold(candidates, 0, 255, cv2.THRESH_BINARY)
    
    # 4. Size Filtering
    min_pixels = int(brain_area * min_tumor_frac)
    max_pixels = int(brain_area * max_tumor_frac)
    print(f"Size range: {min_pixels}-{max_pixels}px")
    
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tumor_candidates)
    valid_tumors = []
    
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_pixels <= area <= max_pixels:
            contour, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contour:
                perimeter = cv2.arcLength(contour[0], True)
                circularity = (4 * np.pi * area) / (perimeter**2 + 1e-5)
                if circularity >= circularity_thresh:
                    valid_tumors.append({
                        'area': area,
                        'circularity': circularity,
                        'intensity': np.mean(img[labels == i])
                    })
                    print(f"Candidate {i}: Area={area}px, Circularity={circularity:.2f}, Intensity={np.mean(img[labels == i]):.1f}")

    # 5. Visualization with Debug Info
    plt.figure(figsize=(12,6))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('Original MRI')
    
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(1, n_labels):
        color = (0,255,0) if min_pixels <= stats[i, cv2.CC_STAT_AREA] <= max_pixels else (0,0,255)
        cv2.rectangle(output, (stats[i,0], stats[i,1]), 
                     (stats[i,0]+stats[i,2], stats[i,1]+stats[i,3]), color, 1)
    
    plt.subplot(122), plt.imshow(output)
    plt.title(f'Candidates (Green=Valid)\nSize:{min_pixels}-{max_pixels}px, Circ>={circularity_thresh}')
    plt.savefig('tumor_debug.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    return {
        'tumor_count': len(valid_tumors),
        'tumors': valid_tumors,
        'debug_info': {
            'brain_area': brain_area,
            'intensity_threshold': tumor_intensity_thresh,
            'size_range': (min_pixels, max_pixels)
        }
    }


import os
import glob

image_dir = "thismeningioma"
image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))

tumor_detected_count = 0

for image_path in image_paths:
    print(f"\nProcessing {os.path.basename(image_path)}...")
    results = detect_tumors(image_path,
                            min_tumor_frac=0.01,
                            max_tumor_frac=0.1,
                            intensity_thresh=0.75,
                            circularity_thresh=0.5)

    if results is None:
        print("Error: detect_tumors() returned None.")
        continue

    print(f"Found {results['tumor_count']} tumor(s)")

    if results['tumor_count'] > 0:
        tumor_detected_count += 1
        for i, tumor in enumerate(results['tumors']):
            print(f"  Tumor {i+1}: {tumor}")

print(f"\nSummary: Tumors detected in {tumor_detected_count} out of {len(image_paths)} images.")







'''
# Example usage with suggested starting values
results = detect_tumors("thisnotumor/Tr-no_0332.jpg",
#results = detect_tumors("thismeningioma/Tr-me_0030.jpg",
                       min_tumor_frac=0.01,  # Try 0.005 for very small tumors
                       max_tumor_frac=0.2,    # Try 0.3 for diffuse tumors
                       intensity_thresh=0.85, # Try 0.7 for low-contrast tumors
                       circularity_thresh=0.4) # Try 0.3 for irregular shapes

print("\nFinal Results:")
print(f"Found {results['tumor_count']} tumor(s)")
for i, tumor in enumerate(results['tumors']):
    print(f"Tumor {i+1}: {tumor['area']}px, Circularity={tumor['circularity']:.2f}")
'''

