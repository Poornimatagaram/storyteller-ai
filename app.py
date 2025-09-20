# app.py
import os
import json
import requests
import subprocess
import streamlit as st
import tempfile # Use tempfile for better cross-platform compatibility
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from requests.exceptions import RequestException # For better error handling

# --- Main App Interface ---
st.set_page_config(page_title="AI Storyteller", page_icon="🎬", layout="centered")
st.title("🎬 AI Storyteller Video Generator")
st.markdown("Turn your ideas into short videos. Enter a topic below and let the AI do the rest!")

# User input for the video topic
video_topic = st.text_input("Enter a video topic:", placeholder="e.g., A robot learning to paint a sunset")

# --- Core Video Generation Logic ---
def generate_video(topic, openai_api_key):
    """
    The main function to generate the video from a topic.
    This function will be called when the button is pressed.
    """
    with st.status("🎬 Let's make a movie...", expanded=True) as status:
        try:
            # --- 1. Initialize Clients ---
            langchain_client = ChatOpenAI(model="gpt-4o", temperature=0.7, api_key=openai_api_key)
            openai_client = OpenAI(api_key=openai_api_key)
            
            # Use a temporary directory to store files for this session
            with tempfile.TemporaryDirectory() as temp_dir:
                status.update(label="🤖 Generating script...")
                
                # --- 2. Create the Prompt Template for the Script ---
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

                # --- 3. Run the Script Generation Chain ---
                script_generation_chain = prompt_template | langchain_client
                response = script_generation_chain.invoke({"topic": topic})
                script_text_response = response.content

                # --- 4. Parse the Script and Generate Media ---
                json_start_index = script_text_response.find('{')
                json_end_index = script_text_response.rfind('}') + 1
                json_only_string = script_text_response[json_start_index:json_end_index]
                storyboard = json.loads(json_only_string)
                scenes = storyboard.get("scenes", [])

                intermediate_video_files = []
                for i, scene in enumerate(scenes):
                    scene_number = i + 1
                    
                    # --- Image Generation ---
                    status.update(label=f"🎨 Generating image for scene {scene_number}...")
                    image_prompt = scene.get("image_prompt")
                    image_response = openai_client.images.generate(model="dall-e-3", prompt=image_prompt, size="1024x1024", quality="standard", n=1)
                    image_url = image_response.data[0].url
                    
                    # Download the image
                    image_data = requests.get(image_url, timeout=30).content
                    image_file_path = os.path.join(temp_dir, f"scene_{scene_number}.png")
                    with open(image_file_path, "wb") as f: f.write(image_data)
                    st.image(image_file_path, caption=f"Scene {scene_number}")

                    # --- Audio Generation ---
                    status.update(label=f"🎤 Generating audio for scene {scene_number}...")
                    voiceover_text = scene.get("voiceover_text")
                    audio_response = openai_client.audio.speech.create(model="tts-1", voice="alloy", input=voiceover_text)
                    audio_file_path = os.path.join(temp_dir, f"scene_{scene_number}.mp3")
                    audio_response.stream_to_file(audio_file_path)

                    # --- Create Intermediate Video Clip for the Scene ---
                    status.update(label=f"🎞️ Creating video clip for scene {scene_number}...")
                    output_video_path = os.path.join(temp_dir, f"scene_{scene_number}.mp4")
                    ffmpeg_command = ["ffmpeg", "-loop", "1", "-i", image_file_path, "-i", audio_file_path, "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", output_video_path, "-y"]
                    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    intermediate_video_files.append(output_video_path)
                
                # --- 5. Assemble the Final Video ---
                status.update(label="🎬 Assembling the final video...")
                filelist_path = os.path.join(temp_dir, "filelist.txt")
                with open(filelist_path, "w") as f:
                    for file in intermediate_video_files:
                        f.write(f"file '{os.path.basename(file)}'\n")

                final_video_path = os.path.join(temp_dir, "final_video.mp4")
                concat_command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", filelist_path, "-c", "copy", final_video_path, "-y"]
                subprocess.run(concat_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Return the final video path so it can be displayed
                return final_video_path
        
        except (RequestException, json.JSONDecodeError, IndexError, subprocess.CalledProcessError, Exception) as e:
            st.error(f"An error occurred during video generation: {e}", icon="🚨")
            status.update(label="Error!", state="error")
            return None

# The button to start the process
if st.button("Generate Video", type="primary"):
    # Check for API key first
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("OpenAI API key not found. Please add it to your Streamlit secrets.", icon="🔒")
    elif not video_topic:
        st.warning("Please enter a topic for the video.", icon="⚠️")
    else:
        # Get the API key from secrets
        api_key = st.secrets["OPENAI_API_KEY"]
        final_video_path = generate_video(video_topic, api_key)
        
        if final_video_path:
            st.success("🎉 Video Generated Successfully!")
            # Read the video file into memory to display it
            video_file = open(final_video_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)