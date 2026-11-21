import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
from skimage import color

# ----------------------------------
# ORB
# ----------------------------------
def extract_orb_features(image_gray, n_features=500):
	orb = cv2.ORB_create(nfeatures=n_features)
	keypoints, descriptors = orb.detectAndCompute(image_gray, None)
	return keypoints, descriptors

# ----------------------------------
# SIFT
# ----------------------------------
def extract_sift_features(image_gray):
	sift = cv2.SIFT_create()
	keypoints, descriptors = sift.detectAndCompute(image_gray, None)
	return keypoints, descriptors

# ----------------------------------
# LBP (Textura local)
# ----------------------------------
def extract_lbp_features(image_gray, P=8, R=1):
	lbp = local_binary_pattern(image_gray, P, R, method="uniform")
	n_bins = int(lbp.max() + 1)
	hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
	hist = hist.astype("float")
	hist /= (hist.sum() + 1e-7)	
	return hist

# ----------------------------------
# HOG (Forma local)
# ----------------------------------
def extract_hog_features(image_gray):
	features, hog_image = hog(
		image_gray,
		orientations=9,
		pixels_per_cell=(8, 8),
		cells_per_block=(2, 2),
		block_norm="L2-Hys",
		visualize=True
	)
	return features, hog_image


def extract_local_descriptors(image_path, method="ORB"):
	img = cv2.imread(image_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if method == "ORB":
		kp, desc = extract_orb_features(gray)
		return kp, desc, img
	elif method == "SIFT":
		kp, desc = extract_sift_features(gray)
		return kp, desc, img
	elif method == "LBP":
		lbp_hist = extract_lbp_features(gray)
		return lbp_hist, img
	elif method == "HOG":
		hog_feat, hog_img = extract_hog_features(gray)
		return hog_feat, hog_img, img
	else:
		raise ValueError("Método no soportado. Usa ORB, SIFT, LBP o HOG.")

image_path = "tulipan.jpeg"

# ORB
kp, desc, img = extract_local_descriptors(image_path, "ORB")
print("ORB -> keypoints:", len(kp), " descriptor shape:", desc.shape)

# SIFT
kp_s, desc_s, _ = extract_local_descriptors(image_path, "SIFT")
print("SIFT -> keypoints:", len(kp_s), " descriptor shape:", desc_s.shape)

# LBP
lbp_hist, _ = extract_local_descriptors(image_path, "LBP")
print("LBP -> hist shape:", lbp_hist.shape)

# HOG
hog_feat, hog_img, _ = extract_local_descriptors(image_path, "HOG")
print("HOG -> feature vector shape:", hog_feat.shape)

# --- VISUALIZACIÓN EXTERNA ---

# Visualizar ORB keypoints
img_kp = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imshow("ORB Keypoints", img_kp)

# Visualizar HOG
# Reescalamos hog_image para que sea visible en cv2.imshow (0-255)
hog_vis = (hog_img * 255).astype("uint8")
cv2.imshow("HOG", hog_vis)

# Mantener ventanas abiertas
cv2.waitKey(0)
cv2.destroyAllWindows()