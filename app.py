import streamlit as st
from PIL import Image
from image_backend import (
    get_device_info,
    preprocess_image,
    transform_image,
    calculate_mse,
    generate_difference_map,
)

st.set_page_config(
    page_title="AI Image Variation & Reconstruction Studio",
    page_icon="🖼️",
    layout="wide",
)

st.title("AI Image Variation & Reconstruction Studio")
st.caption("Transform an image with a pretrained AI model, then measure how the output differs from the original.")

with st.sidebar:
    st.header("Transformation settings")
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg"],
        help="Choose a clear image in PNG or JPEG format.",
    )
    prompt = st.text_area(
        "Transformation prompt",
        value="turn this image into a detailed watercolor painting",
        help="Describe how you want the AI to transform the uploaded image.",
    )
    negative_prompt = st.text_area(
        "Negative prompt (optional)",
        value="blurry, low quality, distorted, artifacts",
    )
    strength = st.slider(
        "Transformation strength",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Lower values tend to preserve more of the input; higher values allow larger changes.",
    )
    guidance_scale = st.slider(
        "Prompt guidance scale",
        min_value=1.0,
        max_value=12.0,
        value=7.5,
        step=0.5,
    )
    steps = st.slider("Inference steps", min_value=5, max_value=40, value=20, step=1)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    run_button = st.button("✨ Transform image", type="primary", use_container_width=True)

st.info(
    "The first run downloads model weights from Hugging Face and may take several minutes. "
    "A compatible GPU is recommended; CPU inference can be very slow."
)
st.caption(f"Runtime device: {get_device_info()}")

if uploaded_file is None:
    st.markdown("### Get started")
    st.write("1. Upload an image in the sidebar.")
    st.write("2. Describe the transformation you want.")
    st.write("3. Adjust strength and inference settings, then select **Transform image**.")
    st.write("4. Compare the generated result and inspect the MSE and difference map.")
else:
    try:
        original = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open this image. Please upload a valid PNG or JPEG. Details: {exc}")
        st.stop()

    st.subheader("Input image")
    st.image(original, caption=f"Original image · {original.width} × {original.height}", use_container_width=True)

    if run_button:
        if not prompt.strip():
            st.warning("Please enter a transformation prompt.")
        else:
            try:
                with st.spinner("Preprocessing image and loading the pretrained model if needed..."):
                    model_image = preprocess_image(original)
                with st.spinner("Running image-to-image inference. This may take a while..."):
                    generated = transform_image(
                        image=model_image,
                        prompt=prompt.strip(),
                        negative_prompt=negative_prompt.strip() or None,
                        strength=float(strength),
                        guidance_scale=float(guidance_scale),
                        num_inference_steps=int(steps),
                        seed=int(seed),
                    )
                st.session_state["original_result"] = original.copy()
                st.session_state["generated_result"] = generated
                st.session_state["run_settings"] = {
                    "Prompt": prompt.strip(),
                    "Strength": strength,
                    "Guidance scale": guidance_scale,
                    "Inference steps": steps,
                    "Seed": seed,
                }
            except Exception as exc:
                st.error(
                    "The transformation could not be completed. Check your internet connection, "
                    "model access, installed dependencies, and available RAM/VRAM. "
                    f"Technical details: {exc}"
                )

if "generated_result" in st.session_state:
    original_result = st.session_state["original_result"]
    generated_result = st.session_state["generated_result"]

    st.divider()
    st.subheader("Original vs. AI-transformed image")
    left, right = st.columns(2)
    with left:
        st.image(original_result, caption="Original image", use_container_width=True)
    with right:
        st.image(generated_result, caption="AI-transformed image", use_container_width=True)

    mse, comparison_size = calculate_mse(original_result, generated_result)
    diff_map = generate_difference_map(original_result, generated_result)

    metric_col, info_col = st.columns([1, 2])
    with metric_col:
        st.metric("Mean Squared Error (MSE)", f"{mse:,.2f}")
    with info_col:
        st.write(
            f"MSE was calculated after resizing both images to **{comparison_size[0]} × "
            f"{comparison_size[1]}** pixels. For 8-bit RGB values, MSE ranges from 0 to 65,025; "
            "zero means the aligned pixel values are identical."
        )

    st.subheader("Pixel difference map")
    st.image(
        diff_map,
        caption="Brighter areas indicate larger absolute pixel differences (visualization normalized for display).",
        use_container_width=True,
    )
    st.warning(
        "MSE measures pixel-level difference, not whether the AI output is more attractive, "
        "more accurate, or semantically better. Color changes and small shifts can increase MSE."
    )

    settings = st.session_state.get("run_settings", {})
    with st.expander("Settings used for this result"):
        for key, value in settings.items():
            st.write(f"**{key}:** {value}")

    from io import BytesIO
    buffer = BytesIO()
    generated_result.save(buffer, format="PNG")
    st.download_button(
        "Download transformed image",
        data=buffer.getvalue(),
        file_name="ai_transformed_image.png",
        mime="image/png",
    )

st.divider()
st.caption("Academic prototype · Python · Streamlit · PyTorch · Hugging Face Diffusers")
