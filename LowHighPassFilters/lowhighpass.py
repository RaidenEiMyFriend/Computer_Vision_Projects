import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image from the file system
image_path = 'image.jpg'
image = cv2.imread(image_path)

# Convert the image to RGB from BGR

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply a GaussianBlur as a low-pass filter
blurred_image = cv2.GaussianBlur(image_rgb, (9, 9), 0)

# Load image in grayscale
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded correctly
if gray_image is None:
    print("Need a image")
else:
    # Create a high pass kernel (edge detection filter)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])

    # Apply the kernel to the image using the filter2D function
    high_pass_image = cv2.filter2D(gray_image, -1, kernel)

    # Show the original image and the image with the high pass filter
    plt.figure(figsize=(20,10))
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(blurred_image)
    plt.title('Low-Pass Filtered Image')
    plt.axis('off')


    plt.subplot(2, 3, 3)
    plt.imshow(high_pass_image, cmap='gray')
    plt.title('High-Pass Filtered Image')
    plt.axis('off')

    plt.show()        