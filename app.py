import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import cv2
import numpy as np

def detect_clothing(image_path, target_label):
    """
    Detects a specific clothing type in an image and returns its bounding boxes.

    Args:
        image_path (str): Path to the input image.
        target_label (str): The clothing type to detect ('shirt', 't-shirt', 'pants', 'dress', 'coat').

    Returns:
        list: A list of bounding boxes [(x, y, w, h)] for the target clothing type.
    """
    try:
        model_config = "yolov8n.yaml"  # Replace with your model's config
        model_weights = "yolov8n.pt"    # Replace with your model's weights

        net = cv2.dnn.readNet(model_weights, model_config)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not open or find the image at {image_path}")
            return []

        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        relevant_class_indices = []

        relevant_labels_map = {
            'shirt': [60],
            't-shirt': [61],
            'pants': [56],
            'dress': [58, 57], # Include skirt as potentially part of a 'dress' outfit
            'coat': [59]
        }

        target_class_ids = relevant_labels_map.get(target_label.lower(), [])

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id in target_class_ids and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    relevant_class_indices.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        final_boxes = []
        if indices is not None:
            for i in indices.flatten():
                final_boxes.append(boxes[i])

        return final_boxes

    except Exception as e:
        print(f"Error in detect_clothing: {e}")
        return []

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    st.set_page_config(page_title="Outfit Room",
                        page_icon=":dress:")
    st.title(":dress: Outfit room")
    st.markdown("*Upload your photo and outfit photo")

    col1, col2 = st.columns(2)

    with col1:
        top_image = st.file_uploader("Upload Your Image", type=["jpg", "jpeg", "png"], key="top")

    with col2:
        bottom_image = st.file_uploader("Upload Outfit Image", type=["jpg", "jpeg", "png"], key="bottom")

    outfit_type = st.selectbox("Select Outfit Type to Circle", ["", "Shirt", "Pants", "Coat", "Dress"])

    if st.button("Submit"):
        if not top_image:
            st.error("Please upload a Your image.")
        if not bottom_image:
            st.error("Please upload a Outfit image.")
        if not outfit_type:
            st.error("Please select an outfit type to circle.")
        elif outfit_type == "":
            st.error("Please select an outfit type to circle.")
        elif top_image and bottom_image and outfit_type:
            st.success(f"Circling '{outfit_type}' in the Outfit Image...")

            top_image_pil = Image.open(top_image)
            bottom_image_pil = Image.open(bottom_image)

            st.image(top_image_pil, caption="Your Image", use_container_width=True)

            # Save the outfit image temporarily for OpenCV
            with open("temp_outfit_image.jpg", "wb") as f:
                f.write(bottom_image.read())

            # Detect the specified clothing type
            bounding_boxes = detect_clothing("temp_outfit_image.jpg", outfit_type)

            # Draw circles around the detected clothing
            img_cv = cv2.imread("temp_outfit_image.jpg")
            if img_cv is not None:
                for x, y, w, h in bounding_boxes:
                    center_x = int(x + w / 2)
                    center_y = int(y + h / 2)
                    radius = int(max(w, h) / 2)
                    cv2.circle(img_cv, (center_x, center_y), radius, (0, 255, 0), 3) # Green circle

                # Convert the OpenCV image back to PIL for Streamlit
                img_pil_circled = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                st.image(img_pil_circled, caption=f"Outfit Image with '{outfit_type}' Circled", use_container_width=True)
            else:
                st.error("Could not load the outfit image for drawing.")

            os.remove("temp_outfit_image.jpg") # Clean up temporary file

            st.write(f"Outfit Type Selected to Circle: {outfit_type}")

if __name__ == "__main__":
    main()