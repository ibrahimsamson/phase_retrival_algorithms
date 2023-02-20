import cv2
import numpy as np

def bit_plane_slicing(img_path, plane):
    # Load image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Extract the specified bit plane
    bit_plane = np.bitwise_and(img, 2 ** plane)
    # Scale the bit plane values to 0-255 range
    bit_plane = (bit_plane / 2 ** plane) * 255
    # Convert bit plane to uint8 data type
    bit_plane = bit_plane.astype(np.uint8)
    # Display the bit plane image
    cv2.imshow("Bit Plane " + str(plane), bit_plane)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


