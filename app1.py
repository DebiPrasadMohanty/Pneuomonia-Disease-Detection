import streamlit as st
from PIL import Image, ImageEnhance
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load the TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_detection_model.h5')
    return model

# Identify the last convolutional layer
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Model does not contain any Conv2D layers.")

# Check image quality
def check_image_quality(image):
    width, height = image.size
    return width >= 256 and height >= 256

# Check if the image is blurry
def is_blurry(image):
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < 100  # Adjust threshold as needed

# Enhance image quality
def enhance_image(image):
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.5)  # Increase brightness
    return image

# Grad-CAM functions
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(img, heatmap, alpha=0.4):
    img = np.array(img.convert("RGB"))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# LIME function
def explain_with_lime(image, model):
    def predict_fn(images):
        images = np.array([img / 255.0 for img in images])
        preds = model.predict(images)
        return preds

    explainer = lime_image.LimeImageExplainer()
    img = np.array(image.convert("RGB").resize((256, 256)))
    explanation = explainer.explain_instance(
        img,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    img_boundry = mark_boundaries(temp / 255.0, mask)
    return img_boundry

# Predict function with Grad-CAM
def predict_with_gradcam(image, model, last_conv_layer_name):
    # Preprocess the image
    img = image.convert("RGB")
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Make prediction
    preds = model.predict(img_array)
    prediction = preds[0][0]
    label = "Pneumonia Detected" if prediction > 0.5 else "No Pneumonia Detected"

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    # Superimpose heatmap on image
    superimposed_img = superimpose_heatmap(img, heatmap)

    return label, superimposed_img, heatmap

# Streamlit front end
st.title("Real-Time Pneumonia Detection with Explanations")

# Image uploader
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Check image quality
    if not check_image_quality(image) or is_blurry(image):
        st.warning("The uploaded image may be of poor quality. Please upload a clearer image.")
        
        # Optionally enhance image and show it
        enhanced_image = enhance_image(image)
        st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
        st.warning("Image has been enhanced. You may proceed with prediction.")
    else:
        # Enhance image
        image = enhance_image(image)
        st.image(image, caption="Enhanced Image", use_column_width=True)

    # Load the TensorFlow model
    model = load_model()
    last_conv_layer_name = get_last_conv_layer(model)

    # Predict and display the result with Grad-CAM
    if st.button("Predict"):
        with st.spinner("Analyzing..."):
            try:
                label, superimposed_img, heatmap = predict_with_gradcam(image, model, last_conv_layer_name)
                st.success(label)
                
                # Display Grad-CAM
                st.subheader("Grad-CAM Explanation")
                st.image(superimposed_img, caption="Grad-CAM", use_column_width=True)
                
                # Optionally, display the heatmap separately
                st.subheader("Heatmap")
                fig, ax = plt.subplots()
                ax.imshow(heatmap, cmap='viridis')
                ax.axis('off')
                st.pyplot(fig)
                
                # Optionally, provide LIME explanation
                if st.checkbox("Show LIME Explanation"):
                    lime_img = explain_with_lime(image, model)
                    st.subheader("LIME Explanation")
                    st.image(lime_img, caption="LIME", use_column_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
