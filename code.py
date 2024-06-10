import cv2
import numpy as np

def apply_custom_artistic_filters(input_image_path, output_prefix):
    # Read image
    img = cv2.imread(input_image_path)

    # Custom Filter 1: Grayscale
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_prefix + "_grayscale.jpg", grayscale_img)

    # Custom Filter 2: Sepia
    sepia_img = apply_sepia(img)
    cv2.imwrite(output_prefix + "_sepia.jpg", sepia_img)

    # Custom Filter 3: Custom Edge Detection
    edge_img = custom_edge_detection(img)
    cv2.imwrite(output_prefix + "_custom_edge_detection.jpg", edge_img)

def apply_sepia(image):
    # Sepia transformation matrix
    sepia_matrix = np.array([[0.393, 0.769, 0.189],
                            [0.349, 0.686, 0.168],
                            [0.272, 0.534, 0.131]])

    sepia_img = cv2.transform(image, sepia_matrix)
    sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)

    return sepia_img

def custom_edge_detection(image):
    # Custom edge detection kernel
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])

    # Apply convolution
    edge_img = cv2.filter2D(image, -1, kernel)
    edge_img = np.clip(edge_img, 0, 255).astype(np.uint8)

    return edge_img

if __name__ == "__main__":
    # Replace 'input_image.jpg' with the path to your image
    input_image_path = 'input_image.jpg'
    img = cv2.imread(input_image_path)

if img is None:
    print(f"Error: Unable to load the image from {input_image_path}")
else:
    # Continue with the image processing


    # Specify an output prefix for the result images
    output_prefix = 'output_image'

    apply_custom_artistic_filters(input_image_path, output_prefix)
