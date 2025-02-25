import streamlit as st
import os
import yt_dlp
import ffmpeg
import torch
import cv2
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from accelerate import Accelerator
from transformers import BlipProcessor, BlipForConditionalGeneration

import os
ffmpeg_path = '/mount/src/takeshi/bin/ffmpeg/ffmpeg'
ffprobe_path = '/mount/src/takeshi/bin/ffmpeg/ffprobe'

# Check if the files are executable
st.write(f"FFmpeg executable exists: {os.access(ffmpeg_path, os.X_OK)}")
st.write(f"FFprobe executable exists: {os.access(ffprobe_path, os.X_OK)}")

def load_model(model_choice):
    model_map = {
        "Anything-V5": "stablediffusionapi/anything-v5",
        "DeepGHS AnimeFull-Latest": "deepghs/animefull-latest",
        "AnimeFull-Final-Pruned": "jianghushinian/animefull-final-pruned",
    }
    model_id = model_map.get(model_choice, "stablediffusionapi/anything-v5")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, safety_checker=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.to(device)
    accelerator = Accelerator()
    pipe = accelerator.prepare(pipe)

    if device == "cuda":
        torch.cuda.empty_cache()

    return pipe

@st.cache_resource
def get_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(image_path, processor, model):
    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")
    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

def create_anime_prompt(caption, extra_prompt=""):
    return f"{caption}, {extra_prompt}" if extra_prompt else caption

def download_video(video_url, output_path):
    st.info("Downloading video...")
    ydl_opts = {'format': 'best', 'outtmpl': output_path}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        st.success("Video downloaded successfully!")
    except Exception as e:
        st.error(f"Error downloading video: {e}")
    return output_path

def extract_key_frames(video_path, output_folder, frame_interval=30):
    st.info("Extracting key frames...")
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count, saved_frame_count = 0, 1
    success, frame = cap.read()
    while success:
        if frame_count % frame_interval == 0:
            cv2.imwrite(os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg"), frame)
            saved_frame_count += 1
        success, frame = cap.read()
        frame_count += 1
    cap.release()
    st.success(f"Frames extracted successfully! {saved_frame_count - 1} frames saved.")
    return output_folder

def convert_to_anime(input_folder, output_folder, pipe, processor, model, strength=0.5, guidance=10.0, extra_prompt=""):
    st.info("Converting frames...")
    os.makedirs(output_folder, exist_ok=True)
    for frame in os.listdir(input_folder):
        input_path = os.path.join(input_folder, frame)
        output_path = os.path.join(output_folder, frame)
        caption = generate_caption(input_path, processor, model)
        anime_prompt = create_anime_prompt(caption, extra_prompt)
        st.write(f"Generated Prompt for {frame}: {anime_prompt}")
        init_image = Image.open(input_path).convert("RGB").resize((432, 240))
        result = pipe(prompt=anime_prompt, image=init_image, strength=strength, guidance_scale=guidance).images[0]
        result.save(output_path)
    st.success("Anime conversion completed!")
    return output_folder

def create_video_from_frames(input_folder, output_video, fps=24):
    st.info("Generating video from frames...")
    input_pattern = os.path.join(input_folder, "frame_%05d.jpg")
    ffmpeg.input(input_pattern, framerate=fps).output(output_video, vcodec='libx264', pix_fmt='yuv420p').run()
    st.success("Video created successfully!")
    return output_video

# Streamlit UI
st.title("Anime Scene Generator")

project_name = st.text_input("Enter Project Name:")
model_choice = st.selectbox("Select AI Model:", ["Anything-V5", "DeepGHS AnimeFull-Latest", "AnimeFull-Final-Pruned" ])
pipe = load_model(model_choice)
processor, model = get_blip()

video_url = st.text_input("Enter Video URL for Scene:", value="https://www.youtube.com/watch?v=8DajVKAkL50")
extra_prompt = st.text_input("Enter additional prompt keywords:")
frame_interval = st.slider("Select frame extraction interval:", min_value=1, max_value=240, value=180, step=1)
strength = st.slider("Select strength (0.0 to 1.0):", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
st.markdown("""
    **Strength Explanation:**
    - Lower strength (0.0) results in a more subtle transformation, keeping more of the original image.
    - Higher strength (1.0) results in a more intense transformation, making the image more anime-like.
""")
guidance = st.slider("Select guidance scale (1.0 to 20.0):", min_value=1.0, max_value=20.0, value=10.0, step=0.1)
st.markdown("""
    **Guidance Scale Explanation:**
    - Lower guidance scale (e.g., 1.0) makes the generated image more faithful to the prompt but can reduce creativity.
    - Higher guidance scale (e.g., 20.0) gives more freedom to the model, allowing it to create more diverse and imaginative results.
""")

if video_url and project_name:
    project_folder = os.path.join("projects", project_name)
    os.makedirs(project_folder, exist_ok=True)

    if st.button("Convert"):
        stop_process = st.button("Stop")
        if stop_process:
            st.warning("Process stopped by user.")
            st.stop()

        video_path = download_video(video_url, os.path.join(project_folder, "source_video.mp4"))
        frames_folder = extract_key_frames(video_path, os.path.join(project_folder, "output_frames"), frame_interval)
        source_video = create_video_from_frames(frames_folder, os.path.join(project_folder, "source_scene.mp4"))
        anime_folder = convert_to_anime(frames_folder, os.path.join(project_folder, "anime_frames"), pipe, processor, model, strength=strength, guidance=guidance, extra_prompt=extra_prompt)
        output_video = create_video_from_frames(anime_folder, os.path.join(project_folder, "anime_scene.mp4"))
        st.video(output_video)
        st.stop()
