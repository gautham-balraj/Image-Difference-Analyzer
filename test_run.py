import cv2
import numpy as np
import sys
from skimage.metrics import structural_similarity

def find_image_difference(before, after):
    """
    Find the differences between two input images.

    Args:
        before (ndarray): The first input image.
        after (ndarray): The second input image.

    Returns:
        tuple: A tuple containing the following:
            - result_before (ndarray): The first input image with differences highlighted.
            - result_after (ndarray): The second input image with differences highlighted.
            - diff (ndarray): The difference image.
            - diff_box (ndarray): The difference image with bounding boxes drawn around differences.
            - mask (ndarray): Mask representing the areas of difference.
            - filled_after (ndarray): The second input image with areas of difference filled.
            - similarity_score (float): The similarity score between the two images.
    """

    # Convert images to grayscale
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (similarity_score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(similarity_score * 100))

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv2.merge([diff, diff, diff])

    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    mask = np.zeros(before.shape, dtype='uint8')
    filled_after = after.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.drawContours(mask, [c], 0, (255,255,255), -1)
            cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

    return before, after, diff, diff_box, mask, filled_after, similarity_score

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <image1_path> <image2_path>")
        sys.exit(1)

    # Load images
    image1_path = sys.argv[1]
    image2_path = sys.argv[2]
    before = cv2.imread(image1_path)
    after = cv2.imread(image2_path)

    # Find differences and get similarity score
    result_before, result_after, diff, diff_box, mask, filled_after, similarity_score = find_image_difference(before, after)

    # Display result
    #cv2.imshow('before', result_before)
    cv2.imshow('after', result_after)
    #cv2.imshow('diff', diff)
    #cv2.imshow('diff_box', diff_box)
    #cv2.imshow('mask', mask)
    #cv2.imshow('filled after', filled_after)

    # Save the result_after image as output
    result_after_path = "result_after.png"  # Change the extension if needed
    cv2.imwrite(result_after_path, result_after)
    print(f"Result after image saved as: {result_after_path}")

    cv2.waitKey()
