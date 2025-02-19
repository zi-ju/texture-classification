import skimage.feature as skf
import numpy as np


# GLCM Feature Extraction
def extract_glcm_features(image):
    image = (image * 255).astype(np.uint8)
    distances = [1, 2, 3, 4, 5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
    glcm = skf.graycomatrix(image, distances, angles, symmetric=True, normed=True)

    features = []
    features.append(skf.graycoprops(glcm, 'contrast').mean())
    features.append(skf.graycoprops(glcm, 'correlation').mean())
    features.append(skf.graycoprops(glcm, 'energy').mean())
    features.append(skf.graycoprops(glcm, 'homogeneity').mean())

    return features


# LBP Feature Extraction
def extract_lbp_features(image, radius=3, points=8):
    image = (image * 255).astype(np.uint8)
    lbp = skf.local_binary_pattern(image, points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 3), density=True)
    return hist.tolist()


# Use GLCM or LBP to extract features
def extract_features(images, method):
    feature_vectors = []
    for img in images:
        if method == "glcm":
            features = extract_glcm_features(img)
        elif method == "lbp":
            features = extract_lbp_features(img)
        else:
            raise ValueError("Invalid method. Choose 'glcm' or 'lbp'.")
        feature_vectors.append(features)
    return np.array(feature_vectors)
