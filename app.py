import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import cv2
import numpy as np

def detect_clothing(image_path):
    """
    Detects various clothing types in an image and returns a list of unique labels.

    Args:
        image_path (str): Path to the input image.

    Returns:
        list: A list of unique detected clothing types (e.g., ['shirt', 'pants']).
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

        detected_labels = set()
        confidences = []
        boxes = []

        relevant_labels_map = {
            0: 'person',
            56: 'pants',
            57: 'skirt',
            58: 'dress',
            59: 'coat',
            60: 'shirt',
            61: 't-shirt'
        }
        target_labels = ['shirt', 't-shirt', 'pants', 'skirt', 'dress', 'coat']

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                label = relevant_labels_map.get(class_id)

                if label in target_labels and confidence > 0.5:
                    detected_labels.add(label)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        final_detected_labels = set()
        if indices is not None:
            for i in indices.flatten():
                # You could potentially get the label based on the class ID here if needed
                # For now, we rely on the 'detected_labels' set
                pass

        return list(detected_labels)

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

    outfit_type_selection = st.selectbox("Select Expected Outfit Type", ["", "Top", "Bottom", "One Piece"])

    if st.button("Submit"):
        if not top_image:
            st.error("Please upload a Your image.")
        if not bottom_image:
            st.error("Please upload a Outfit image.")
        if not outfit_type_selection:
            st.error("Please select the expected outfit type.")
        elif outfit_type_selection == "":
            st.error("Please select the expected outfit type.")
        elif top_image and bottom_image and outfit_type_selection:
            st.success(f"Analyzing Outfit Image for type: '{outfit_type_selection}'...")

            top_image_pil = Image.open(top_image)
            st.image(top_image_pil, caption="Your Image", use_container_width=True)

            st.subheader("Outfit Image:")
            st.image(bottom_image, caption="Outfit Image", use_container_width=True)

            # Save the outfit image temporarily for OpenCV
            with open("temp_outfit_image.jpg", "wb") as f:
                f.write(bottom_image.read())

            # Detect clothing types in the outfit image
            detected_clothes = detect_clothing("temp_outfit_image.jpg")
            os.remove("temp_outfit_image.jpg") # Clean up temporary file

            st.subheader("Detected Clothing in Outfit Image:")
            if detected_clothes:
                for cloth_type in detected_clothes:
                    st.markdown(f"- {cloth_type.capitalize()}")

                # Logic to extract outfit type based on detected clothes and user selection
                extracted_outfit_type = "Unknown"
                if outfit_type_selection == "Top":
                    if any(item in ['shirt', 't-shirt'] for item in detected_clothes):
                        extracted_outfit_type = "Top"
                elif outfit_type_selection == "Bottom":
                    if any(item in ['pants', 'skirt'] for item in detected_clothes):
                        extracted_outfit_type = "Bottom"
                elif outfit_type_selection == "One Piece":
                    if 'dress' in detected_clothes:
                        extracted_outfit_type = "One Piece"

                st.subheader("Extracted Outfit Type:")
                st.markdown(f"Based on the detected clothing and your selection, the extracted outfit type is: **{extracted_outfit_type}**")

            else:
                st.info("Could not detect any relevant clothing items in the outfit image to extract the type.")

if __name__ == "__main__":
    main()