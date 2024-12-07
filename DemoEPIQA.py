import cv2
import numpy as np
from scipy.ndimage import median_filter
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

# Preprocessing Step: Median filtering and unsharp masking
def preprocess_image(image, kernel_size=3):
    """
    Preprocess the input image:
    - Median filter: Reduces noise while preserving edges.
    - Unsharp masking: Enhances edges by subtracting a blurred version of the image.
    """
    filtered_image = median_filter(image, size=kernel_size)
    blurred = cv2.GaussianBlur(filtered_image, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(filtered_image, 1.5, blurred, -0.5, 0)
    return unsharp_image

# Feature Extraction: Compute edge orientations within each block
def compute_edge_orientation(block):
    """
    Calculate edge orientation within a block:
    - Use Sobel gradients to compute the orientation of edges.
    - Count edges with horizontal and vertical orientations.
    """
    gradients_x = cv2.Sobel(block, cv2.CV_64F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(block, cv2.CV_64F, 0, 1, ksize=3)
    orientation = np.arctan2(gradients_y, gradients_x) * (180 / np.pi)
    vertical_edges = np.sum((orientation > 45) & (orientation < 135))
    horizontal_edges = np.sum((orientation < -45) | (orientation > -135))
    return vertical_edges + horizontal_edges

# Feature Extraction: Extract edge features from image blocks
def compute_edge_features(image, block_size=8):
    """
    Extract edge features for each block:
    Features include:
    - Edge Density (ED): Proportion of edge pixels in the block.
    - Edge Length Average (ELA): Average length of detected edges.
    - Gray Level Region (GLR): Number of unique intensity levels in the block.
    - Number of Edge Pixels (NEP): Total count of edge pixels.
    - Edge Orientation (EO): Count of vertical and horizontal edges.
    """
    edges = cv2.Canny(image, 100, 200)  # Detect edges using the Canny method
    height, width = image.shape
    edge_features = []
    
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = edges[i:i+block_size, j:j+block_size]
            gray_block = image[i:i+block_size, j:j+block_size]
            
            # Extract features
            edge_density = np.sum(block > 0) / block.size
            edge_length = np.sum(block > 0)
            gray_level_region = len(np.unique(gray_block))
            num_edge_pixels = np.sum(block > 0)
            edge_orientation = compute_edge_orientation(block)
            
            # Append features for this block
            edge_features.append([edge_density, edge_length, gray_level_region, num_edge_pixels, edge_orientation])
    
    return np.array(edge_features)

# Compute Euclidean Distance: Measure similarity between feature vectors
def compute_euclidean_distance(features1, features2):
    """
    Compute the Euclidean distance between two sets of feature vectors.
    """
    distances = np.sqrt(np.sum((features1 - features2) ** 2, axis=1))
    return np.mean(distances)

# EPIQA Calculation: Combine edge-based distance and PSNR
def calculate_epiqa(original_image, distorted_image, block_size=8):
    """
    Calculate the EPIQA score:
    - Preprocess images using median filtering and unsharp masking.
    - Extract edge features from both original and distorted images.
    - Compute normalized Euclidean distance for edge features.
    - Combine normalized distance with PSNR (Peak Signal-to-Noise Ratio).
    """
    original_preprocessed = preprocess_image(original_image)
    distorted_preprocessed = preprocess_image(distorted_image)
    original_features = compute_edge_features(original_preprocessed, block_size)
    distorted_features = compute_edge_features(distorted_preprocessed, block_size)
    distance = compute_euclidean_distance(original_features, distorted_features)
    normalized_distance = 1 - (distance / (original_features.shape[0]))
    psnr_value = psnr(original_image, distorted_image)
    psnr_normalized = psnr_value / 50
    epiqa_score = (normalized_distance + psnr_normalized) / 2
    return epiqa_score

# Validation Metrics: Compare predicted scores with subjective MOS
def calculate_validation_metrics(mos_scores, epiqa_scores):
    """
    Calculate validation metrics:
    - Spearman Rank-Order Correlation Coefficient (SROCC)
    - Kendall Rank-Order Correlation Coefficient (KROCC)
    - Pearson Linear Correlation Coefficient (PLCC)
    - Root-Mean-Square Error (RMSE)
    """
    srocc, _ = spearmanr(mos_scores, epiqa_scores)
    krocc, _ = kendalltau(mos_scores, epiqa_scores)
    mos_mean = np.mean(mos_scores)
    epiqa_mean = np.mean(epiqa_scores)
    numerator = np.sum((mos_scores - mos_mean) * (epiqa_scores - epiqa_mean))
    denominator = np.sqrt(np.sum((mos_scores - mos_mean)**2) * np.sum((epiqa_scores - epiqa_mean)**2))
    plcc = numerator / denominator
    rmse = np.sqrt(mean_squared_error(mos_scores, epiqa_scores))
    return {"SROCC": srocc, "KROCC": krocc, "PLCC": plcc, "RMSE": rmse}

# Main Execution Block
if __name__ == "__main__":
    # Load the reference and noisy images
    original = cv2.imread("depth.png", cv2.IMREAD_GRAYSCALE)
    distorted = cv2.imread("noisy.png", cv2.IMREAD_GRAYSCALE)

    if original is None or distorted is None:
        print("Error: Please provide valid image paths.")
    else:
        # Calculate EPIQA score for the image pair
        epiqa_score = calculate_epiqa(original, distorted)
        print(f"EPIQA Score: {epiqa_score:.4f}")

        # Define example MOS (subjective) and EPIQA (objective) scores
        mos_scores = np.array([5, 4, 3, 2, 1])  # Replace with real MOS scores
        epiqa_scores = np.array([4.9, 4.1, 3.0, 2.1, 1.2])  # Replace with calculated EPIQA scores

        # Calculate and print validation metrics
        metrics = calculate_validation_metrics(mos_scores, epiqa_scores)
        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Analyze results if SROCC or KROCC are perfect
        if metrics["SROCC"] == 1.0 or metrics["KROCC"] == 1.0:
            print("\nPerfect rank correlation suggests identical ranks of MOS and EPIQA scores.")
            print("Check if noisy images were assigned unrealistic MOS scores.")
