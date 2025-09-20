# app.py
import os
import io
import json
import zipfile
import requests
import subprocess
import streamlit as st
import tempfile
import random
import string
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from requests.exceptions import RequestException

# --- Page config & CSS ---
st.set_page_config(page_title="AI Storyteller", page_icon="🎬", layout="centered")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(180deg, #0f172a 0%, #071037 100%);
        color: #e6eef8;
    }
    .header {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .title {
        font-size: 28px;
        font-weight: 700;
        margin: 0;
    }
    .subtitle {
        font-size: 13px;
        color: #bcd3ff;
        margin: 0;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 4px 18px rgba(2,6,23,0.6);
        margin-bottom: 12px;
    }
    .btn-primary {
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        color: white;
        padding: 8px 14px;
        border-radius: 10px;
    }
    .small-muted { color: #9fb0d9; font-size: 13px; }
    .scene-caption { font-weight:600; font-size:14px; color:#eaf2ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="header">
        <div style="font-size:40px">🎬</div>
        <div>
            <div class="title">AI Storyteller</div>
            <div class="subtitle">Turn ideas into short cinematic videos — fast and beautifully.</div>
        </div>
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)


st.markdown("Write a short description or topic and press **Generate Video**. Filenames will be generated randomly.")

# --- Inputs & progress placeholder ---
video_topic = st.text_input("Enter a video topic:", placeholder="e.g., A robot learning to paint a sunset")
progress_bar = st.progress(0)

# initialize session state for persistence
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# --- Generate random filename function ---
def random_filename(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# --- Core generation function ---
def generate_video(topic, openai_api_key, expected_scenes=5, ffmpeg_force_overwrite=True):
    """
    Generate video & images, return dict:
      { video_bytes, images: [{name, bytes}], error (or None) }
    """
    try:
        langchain_client = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=openai_api_key)
        openai_client = OpenAI(api_key=openai_api_key)

        with tempfile.TemporaryDirectory() as temp_dir:
            progress_bar.progress(5)

            # Prompt template
            prompt_template = ChatPromptTemplate.from_messages([
                ("system",
                 "You are a helpful assistant that generates short, 2-scene video scripts. "
                 "Your output must be a valid JSON object. "
                 "Each scene object must have an 'image_prompt' and a 'voiceover_text'. "
                 "The 'image_prompt' must be a highly detailed description for an AI image generator, "
                 "aiming for a photorealistic, cinematic style with dramatic lighting. "
                 "Think like a film director specifying a shot."),
                ("human", "Here is the topic: {topic}"),
            ])

            # Run model to get storyboard JSON
            script_generation_chain = prompt_template | langchain_client
            response = script_generation_chain.invoke({"topic": topic})
            script_text_response = response.content

            # Parse JSON robustly
            json_start_index = script_text_response.find('{')
            json_end_index = script_text_response.rfind('}') + 1
            if json_start_index == -1 or json_end_index == -1:
                raise json.JSONDecodeError("Could not find JSON in model response", script_text_response, 0)
            json_only_string = script_text_response[json_start_index:json_end_index]
            storyboard = json.loads(json_only_string)
            scenes = storyboard.get("scenes", [])
            if not scenes:
                raise ValueError("No scenes found in the storyboard JSON.")

            progress_bar.progress(15)

            intermediate_video_files = []
            images_for_download = []

            for i, scene in enumerate(scenes):
                scene_number = i + 1
                progress_bar.progress(15 + int((i / max(1, len(scenes))) * 60))

                # Image generation
                image_prompt = scene.get("image_prompt")
                if not image_prompt:
                    raise ValueError(f"Scene {scene_number} missing 'image_prompt'")
                image_response = openai_client.images.generate(model="dall-e-3", prompt=image_prompt, size="1024x1024", quality="standard", n=1)
                image_url = image_response.data[0].url
                image_data = requests.get(image_url, timeout=30).content
                image_file_path = os.path.join(temp_dir, f"scene_{scene_number}.png")
                with open(image_file_path, "wb") as f:
                    f.write(image_data)

                # collect image bytes for download
                images_for_download.append({"name": f"scene_{scene_number}.png", "bytes": image_data})

                # Audio generation
                voiceover_text = scene.get("voiceover_text", "")
                audio_response = openai_client.audio.speech.create(model="tts-1", voice="alloy", input=voiceover_text)
                audio_file_path = os.path.join(temp_dir, f"scene_{scene_number}.mp3")
                audio_response.stream_to_file(audio_file_path)

                # Create scene video using ffmpeg
                output_video_path = os.path.join(temp_dir, f"scene_{scene_number}.mp4")
                ffmpeg_command = [
                    "ffmpeg",
                    "-loop", "1",
                    "-i", image_file_path,
                    "-i", audio_file_path,
                    "-c:v", "libx264",
                    "-tune", "stillimage",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-pix_fmt", "yuv420p",
                    "-shortest",
                    output_video_path
                ]
                if ffmpeg_force_overwrite:
                    ffmpeg_command.insert(1, "-y")

                proc = subprocess.run(ffmpeg_command, capture_output=True, text=True)
                if proc.returncode != 0:
                    return {"video_bytes": None, "images": images_for_download, "error": f"FFmpeg failed while creating scene {scene_number} (see server logs)."}
                intermediate_video_files.append(output_video_path)

            # Assemble final video
            filelist_path = os.path.join(temp_dir, "filelist.txt")
            with open(filelist_path, "w") as f:
                for file in intermediate_video_files:
                    f.write(f"file '{file}'\n")

            final_video_path = os.path.join(temp_dir, "final_video.mp4")
            concat_command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", filelist_path, "-c", "copy", final_video_path]
            if ffmpeg_force_overwrite:
                concat_command.insert(1, "-y")

            result = subprocess.run(concat_command, capture_output=True, text=True)

            if result.returncode != 0:
                reencode_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", filelist_path, "-c:v", "libx264", "-c:a", "aac", final_video_path]
                reproc = subprocess.run(reencode_cmd, capture_output=True, text=True)
                if reproc.returncode != 0:
                    return {"video_bytes": None, "images": images_for_download, "error": "Final FFmpeg assembly failed (see server logs)."}

            if not os.path.exists(final_video_path):
                return {"video_bytes": None, "images": images_for_download, "error": "Final video not created."}

            with open(final_video_path, "rb") as vf:
                video_bytes = vf.read()

            progress_bar.progress(100)
            return {"video_bytes": video_bytes, "images": images_for_download, "error": None}

    except subprocess.CalledProcessError as e:
        return {"video_bytes": None, "images": [], "error": "FFmpeg error occurred (see server logs)."}
    except (RequestException, json.JSONDecodeError, IndexError, ValueError) as e:
        return {"video_bytes": None, "images": [], "error": f"Generation error: {e}"}
    except Exception as e:
        return {"video_bytes": None, "images": [], "error": f"Unexpected error: {e}"}

# --- Action: Generate ---
if st.button("Generate Video", type="primary"):
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found. Please add it to your Streamlit secrets.", icon="🔒")
    elif not video_topic:
        st.warning("Please enter a topic for the video.", icon="⚠️")
    else:
        api_key = st.secrets["OPENAI_API_KEY"]
        with st.spinner("Generating your cinematic short... ✨"):
            result = generate_video(video_topic, api_key)

        st.session_state["last_result"] = result

        if result["error"]:
            st.error(result["error"])
        else:
            st.success("🎉 Video Generated Successfully!")

# --- Show results ---
if st.session_state.get("last_result"):
    res = st.session_state["last_result"]
    if res.get("error"):
        st.error(res["error"])
    else:
        random_file = random_filename()
        mp4_name = f"{random_file}.mp4"

        st.markdown("### Preview")
        st.video(res["video_bytes"])

        st.download_button(
            label="⬇️ Download Video",
            data=res["video_bytes"],
            file_name=mp4_name,
            mime="video/mp4",
            help="Click to download the generated MP4"
        )

        if res.get("images"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for img in res["images"]:
                    zf.writestr(img["name"], img["bytes"])
            zip_buffer.seek(0)

            st.download_button(
                label="⬇️ Download All Images (ZIP)",
                data=zip_buffer,
                file_name=f"{random_file}_images.zip",
                mime="application/zip",
                help="All generated scene images in a zip"
            )

            st.markdown("### Scenes")
            cols = st.columns(min(3, len(res["images"])))
            for idx, img in enumerate(res["images"]):
                col = cols[idx % len(cols)]
                col.image(img["bytes"], caption=f"{img['name']}", use_container_width=True)

        if st.button("Clear Results"):
            st.session_state["last_result"] = None
            st.experimental_rerun()
