import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_internal_spheres(img_path, min_radius=5, max_radius=50):
    # 1. Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image")
    
    # CORRECTED LINE: Fixed typo in normalization flag
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # 2. Extract brain region (exclude skull)
    _, brain_mask = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5,5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
    
    # 3. Enhance internal structures
    equalized = cv2.equalizeHist(img)
    blurred = cv2.bilateralFilter(equalized, 9, 75, 75)
    
    # 4. Detect spherical candidates
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 
                            dp=1.2, minDist=20,
                            param1=50, param2=30,
                            minRadius=min_radius,
                            maxRadius=max_radius)
    
    # 5. Filter and analyze
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    metrics = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i, (x, y, r) in enumerate(circles[0,:]):
            # Verify the circle is inside brain
            if brain_mask[y,x] == 0:
                continue
                
            # Create ROI mask
            mask = np.zeros_like(img)
            cv2.circle(mask, (x,y), r, 255, -1)
            
            # Calculate metrics
            roi = img[mask > 0]
            intensity_ratio = np.mean(roi)/np.mean(img[brain_mask > 0])
            
            metrics.append({
                'x': x, 'y': y, 'radius': r,
                'intensity_ratio': intensity_ratio,
                'area': np.pi * r**2,
                'circularity': 1.0  # Perfect by definition
            })
            
            # Visualize
            cv2.circle(output, (x,y), r, (0,255,0), 2)
            cv2.putText(output, f"{i+1}", (x-10,y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    # 6. Visualization
    plt.figure(figsize=(18,6))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(132), plt.imshow(brain_mask, cmap='gray'), plt.title('Brain Mask')
    plt.subplot(133), plt.imshow(output), plt.title('Detected Internal Spheres')
    plt.tight_layout()
    plt.savefig('internal_spheres.png', dpi=300)
    plt.close()
    
    return metrics

# Usage
if __name__ == "__main__":
    try:
        metrics = detect_internal_spheres("thisnotumor/Tr-no_0028.jpg", 
        #metrics = detect_internal_spheres("thismeningioma/Tr-me_0037.jpg", 
                                        min_radius=3, 
                                        max_radius=30)
        print(f"Found {len(metrics)} internal spherical regions")
        for i, m in enumerate(metrics):
            print(f"\nSphere {i+1}:")
            print(f"• Center (x,y): ({m['x']}, {m['y']})")
            print(f"• Radius: {m['radius']}px")
            print(f"• Intensity ratio: {m['intensity_ratio']:.2f}x avg brain intensity")
            print(f"• Area: {m['area']:.0f} px²")
    except Exception as e:
        print(f"Error: {str(e)}")
