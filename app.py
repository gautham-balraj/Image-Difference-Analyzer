import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity
def resize_image(image, target_width, target_height):
    """
    Resize the input image to the target dimensions.
    
    Args:
        image (ndarray): The input image.
        target_width (int): The target width for resizing.
        target_height (int): The target height for resizing.
    
    Returns:
        ndarray: The resized image.
    """
    return cv2.resize(image, (target_width, target_height))
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
    # Resize images to have the same dimensions
    max_width = max(before.shape[1], after.shape[1])
    max_height = max(before.shape[0], after.shape[0])
    before_resized = resize_image(before, max_width, max_height)
    after_resized = resize_image(after, max_width, max_height)
    # Convert images to grayscale
    before_gray = cv2.cvtColor(before_resized, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_resized, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (similarity_score, diff) = structural_similarity(before_gray, after_gray, full=True)
    st.write("Image Similarity: {:.4f}%".format(similarity_score * 100))

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

def main():
    st.set_page_config(page_title="Image Difference Analyzer", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
    st.title("Image Difference Analyzer")

    st.sidebar.title("Upload Images")
    uploaded_file1 = st.sidebar.file_uploader("Upload the first image", type=["jpg", "jpeg", "png"])
    uploaded_file2 = st.sidebar.file_uploader("Upload the second image", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        image1 = cv2.imdecode(np.frombuffer(uploaded_file1.read(), np.uint8), 1)
        image2 = cv2.imdecode(np.frombuffer(uploaded_file2.read(), np.uint8), 1)

        result_before, result_after, diff, diff_box, mask, filled_after, similarity_score = find_image_difference(image1, image2)
        if similarity_score * 100 > 90:
            st.write("The images are similar") 
        else :
            st.write("The images are not similar")
       # st.image(result_before, caption="First Image with Differences Highlighted", use_column_width=True)
        st.image(cv2.cvtColor(result_after, cv2.COLOR_BGR2RGB), caption="Result Image", width=300) # Adjust the width as needed
       # st.image(diff, caption="Difference Image", use_column_width=True)
       # st.image(diff_box, caption="Difference Image with Bounding Boxes", use_column_width=True)
       # st.image(mask, caption="Mask Representing Areas of Difference", use_column_width=True)
       # st.image(filled_after, caption="Second Image with Areas of Difference Filled", use_column_width=True)
        #st.write(f"Similarity Score: {similarity_score * 100:.4f}%")

if __name__ == "__main__":
    main()
