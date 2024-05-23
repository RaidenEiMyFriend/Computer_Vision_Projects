import numpy as np
import cv2 as cv
img = cv.imread('husky.jpg', 3)
rows, cols, depth = img.shape

# translation
T= np.float32([[1, 0, 100], [0, 1, 50]])
translation = cv.warpAffine(img, T, (cols, rows))
cv.imshow('translation', translation)
cv.imwrite('husky_translation.jpg', translation)
cv.waitKey(0)
cv.destroyAllWindows()


# scaling
scaling = cv.resize(img, None, fx=1.5, fy=0.5,interpolation=cv.INTER_CUBIC)
cv.imshow('scaling', scaling)
cv.imwrite('husky_scaling.jpg', scaling)
cv.waitKey(0)
cv.destroyAllWindows()

# rotation
rotation = cv.warpAffine(img, cv.getRotationMatrix2D((cols/2, rows/2),-45, 0.8), (cols, rows))
cv.imshow('rotation', rotation)
cv.imwrite('husky_rotation.jpg', rotation)
cv.waitKey(0)
cv.destroyAllWindows()

# mirror 
M = np.float32([[-1, 0, cols], [0, 1, 0], [0, 0, 1]])
mirror = cv.warpPerspective(img, M, (int(cols), int(rows)))
cv.imshow('mirror', mirror)
cv.imwrite('husky_mirror.jpg', mirror)
cv.waitKey(0)
cv.destroyAllWindows()

# shear in x-aris
Sx = np.float32([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]])
shearx = cv.warpPerspective(img, Sx, (int(cols*1.3), int(rows*1.3)))
cv.imshow('shearx', shearx)
cv.imwrite('husky_shearx.jpg', shearx)
cv.waitKey(0)
cv.destroyAllWindows()

# shear in y-aris
Sy = np.float32([[1, 0, 0], [0.3, 1, 0], [0, 0, 1]])
sheary = cv.warpPerspective(img, Sy, (int(cols*1.3), int(rows*1.3)))
cv.imshow('sheary', sheary)
cv.imwrite('husky_sheary.jpg', sheary)
cv.waitKey(0)
cv.destroyAllWindows()

# projective transformation

P = np.float32([[ 1.6, 0.9, 1.3], [ 0.5, 1.2, 3.6], [0.0008,  0.0009,  1]])
projective = cv.warpPerspective(img, P, (int(cols*1.3), int(rows*1.3)))
cv.imshow('projective', projective)
cv.imwrite('husky_projective.jpg', projective)
cv.waitKey(0)
cv.destroyAllWindows()
