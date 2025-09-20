import os
import json
import requests
import subprocess
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

# --- 1. Load Environment Variables & Initialize Clients ---
load_dotenv()
langchain_client = ChatOpenAI(model="gpt-4o", temperature=0.7)
openai_client = OpenAI()
os.makedirs("output", exist_ok=True)

# --- 2. Create the Prompt Template for the Script ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # --- MODIFIED SECTION ---
            # We've added more descriptive instructions to get better image prompts.
            "You are a helpful assistant that generates short, 2-scene video scripts. "
            "Your output must be a valid JSON object. "
            "Each scene object must have an 'image_prompt' and a 'voiceover_text'. "
            "The 'image_prompt' must be a highly detailed description for an AI image generator, "
            "aiming for a photorealistic, cinematic style with dramatic lighting. "
            "Think like a film director specifying a shot.",
            # --- END OF MODIFIED SECTION ---
        ),
        ("human", "Here is the topic: {topic}"),
    ]
)

# --- 3. Create and Run the Script Generation Chain ---
# (The rest of the file is exactly the same as before)
script_generation_chain = prompt_template | langchain_client
video_topic = "A group of friends playing a game of chess in a park."
print("🤖 Generating script... please wait.")
response = script_generation_chain.invoke({"topic": video_topic})
script_text_response = response.content

print("\n✅ Script Generated Successfully!")
print("---------------------------------")
print(script_text_response)

# --- 4. Parse the Script and Generate Media ---
scenes = []
try:
    json_start_index = script_text_response.find('{')
    json_end_index = script_text_response.rfind('}') + 1
    json_only_string = script_text_response[json_start_index:json_end_index]
    
    storyboard = json.loads(json_only_string)
    scenes = storyboard.get("scenes", [])

    print("\n🤖 Generating media for each scene... this may take a minute.")

    for i, scene in enumerate(scenes):
        scene_number = i + 1
        image_prompt = scene.get("image_prompt")
        if image_prompt:
            print(f"🎨 Generating image for scene {scene_number}...")
            image_response = openai_client.images.generate(model="dall-e-3", prompt=image_prompt, size="1024x1024", quality="standard", n=1)
            image_url = image_response.data[0].url
            image_data = requests.get(image_url).content
            image_file_path = os.path.join("output", f"scene_{scene_number}.png")
            with open(image_file_path, "wb") as f: f.write(image_data)
            print(f"✅ Image for scene {scene_number} saved.")

        voiceover_text = scene.get("voiceover_text")
        if voiceover_text:
            print(f"🎤 Generating audio for scene {scene_number}...")
            audio_response = openai_client.audio.speech.create(model="tts-1", voice="alloy", input=voiceover_text)
            audio_file_path = os.path.join("output", f"scene_{scene_number}.mp3")
            audio_response.stream_to_file(audio_file_path)
            print(f"✅ Audio for scene {scene_number} saved.")

except (json.JSONDecodeError, IndexError) as e:
    print(f"❌ Error: Failed to parse the script or generate media. Details: {e}")

# --- 5. Video Assembly using FFmpeg ---
print("\n🎬 Assembling the final video with FFmpeg...")

intermediate_files = []
for i in range(len(scenes)):
    scene_number = i + 1
    image_path = os.path.join("output", f"scene_{scene_number}.png")
    audio_path = os.path.join("output", f"scene_{scene_number}.mp3")
    output_video_path = os.path.join("output", f"scene_{scene_number}.mp4")

    ffmpeg_command = ["ffmpeg", "-loop", "1", "-i", image_path, "-i", audio_path, "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest", output_video_path, "-y"]
    
    subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    intermediate_files.append(output_video_path)
    print(f"✅ Intermediate video for scene {scene_number} created.")

filelist_path = os.path.join("output", "filelist.txt")
with open(filelist_path, "w") as f:
    for file in intermediate_files:
        f.write(f"file '{os.path.basename(file)}'\n")

final_video_path = os.path.join("output", "final_video.mp4")
concat_command = ["ffmpeg", "-f", "concat", "-safe", "0", "-i", filelist_path, "-c", "copy", final_video_path, "-y"]
subprocess.run(concat_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for file in intermediate_files:
    os.remove(file)
os.remove(filelist_path)

print(f"\n🎉 Success! Your video has been created: {final_video_path}")