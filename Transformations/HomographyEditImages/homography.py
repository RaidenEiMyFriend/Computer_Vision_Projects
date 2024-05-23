import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT
from skimage.transform import warp
from skimage.draw import polygon
from skimage.color import rgba2rgb


def matchPics(I1, I2):
    I1_gray = I1
    # Convert I2 to grayscale and then convert it to unsigned byte
    I2_gray = rgb2gray(I2)

    # Initialize SIFT detector
    descriptor_extractor = SIFT()

    # Detect keypoints and extract descriptors for the first image
    descriptor_extractor.detect_and_extract(I1_gray)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    # Detect keypoints and extract descriptors for the second image
    descriptor_extractor.detect_and_extract(I2_gray)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # Match descriptors between the two images
    matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6, cross_check=True)

    # Return matched descriptors, and keypoints of both images
    return matches12, keypoints1, keypoints2


def computeH_ransac(matches, locs1, locs2, num_iterations=1000, inlier_threshold=10):
    num_matches = matches.shape[0]
    max_inliers = 0
    bestH = None
    inliers = None

    def computeH(src_pts, dst_pts):
        # Construct the A matrix for DLT algorithm
        A = []
        for i in range(src_pts.shape[0]):
            x1, y1 = src_pts[i, :]
            x2, y2 = dst_pts[i, :]
            A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
            A.append([0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2])
        A = np.array(A)

        # Perform Singular Value Decomposition (SVD)
        _, _, Vt = np.linalg.svd(A)

        # The homography is the last column of V transposed
        H = Vt[-1].reshape(3, 3)

        # Normalize the homography
        H = H / H[-1, -1]

        return H

    for _ in range(num_iterations):
        # Randomly sample minimal subset of matches
        subset_indices = np.random.choice(num_matches, 4, replace=False)
        subset_matches = matches[subset_indices]

        # Estimate homography using the subset matches
        src_pts = locs1[subset_matches[:, 0], :]
        dst_pts = locs2[subset_matches[:, 1], :]
        H = computeH(src_pts, dst_pts)

        # Compute the transformed points
        src_pts_homogeneous = np.concatenate((locs1[matches[:, 0]], np.ones((num_matches, 1))), axis=1)
        transformed_pts_homogeneous = np.dot(H, src_pts_homogeneous.T)

        # Normalize the points
        transformed_pts = (transformed_pts_homogeneous[:2] / transformed_pts_homogeneous[2]).T
        dst_pts_all = locs2[matches[:, 1]]

        # Calculate distance between the matched points and the transformed points
        distances = np.sqrt(np.sum((transformed_pts - dst_pts_all) ** 2, axis=1))
        current_inliers = distances < inlier_threshold

        # Check if the number of inliers is greater than the previous maximum
        num_current_inliers = np.sum(current_inliers)
        if num_current_inliers > max_inliers:
            max_inliers = num_current_inliers
            bestH = H
            inliers = current_inliers

    # Refine homography using all inliers
    final_matches = matches[inliers]
    src_pts_inliers = locs1[final_matches[:, 0], :]
    dst_pts_inliers = locs2[final_matches[:, 1], :]
    bestH = computeH(src_pts_inliers, dst_pts_inliers)

    # Reorder the rows and columns to offset the coordination converse because of scikit
    bestH = np.array([[bestH[1, 1], bestH[1, 0], bestH[1, 2]],
                      [bestH[0, 1], bestH[0, 0], bestH[0, 2]],
                      [bestH[2, 1], bestH[2, 0], bestH[2, 2]]])

    return bestH, inliers.nonzero()[0]


def compositeH(H, template, img):
    # Determine the shape of the output image
    out_shape = img.shape[:2]

    # Warp the template using the homography matrix H
    warped_template = warp(template, np.linalg.inv(H), output_shape=out_shape)

    # Create a mask for the warped template
    mask = np.zeros_like(template[:, :, 0], dtype=bool)

    # Generate indices for drawing the polygon
    rr, cc = polygon([0, 0, template.shape[0] - 1, template.shape[0] - 1],
                     [0, template.shape[1] - 1, template.shape[1] - 1, 0])

    mask[rr, cc] = True

    # Warp the mask using the homography matrix H
    warped_mask = warp(mask.astype(float), np.linalg.inv(H), output_shape=out_shape)

    # Convert the warped mask to boolean
    warped_mask = warped_mask > 0.5

    # Convert RGBA template to RGB
    if template.shape[2] == 4:
        template = rgba2rgb(template)

    # Combine the warped template and the image using the mask
    composite_img = img.copy()
    composite_img[warped_mask] = warped_template[warped_mask]

    return composite_img
