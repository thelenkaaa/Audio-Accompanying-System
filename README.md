# Audio-Accompanying-System
A system that generates realistic audio for videos by analyzing highlighted objects on the screen.

Overview.
The Audio Accompanying System is designed to enhance videos without sound by generating realistic audio based on the detected objects. 
The system detects objects in video frames, filters out irrelevant ones, and generates appropriate sounds based on how long each object appears on the screen. In the end, all generated audio pieces are combined into a single synchronized track that should match the original video.

How to see Results.
Open Diploma_07_03_2025.ipynb to see how the system works FOR NOW.

Key Features.
✅ Object Detection – Identifies and tracks objects in video frames.
✅ Smart Sound Filtering – Uses an LLM to filter out objects that don’t have recognizable sounds (e.g., umbrellas, traffic lights).
✅ AI-Powered Sound Generation – Converts detected objects into text prompts for an Audio Model to generate corresponding sounds.
✅ Time-Based Synchronization – Calculates how long each object is visible and generates audio of appropriate length.
✅ Final Audio Merging – Combines all individual sounds into one cohesive track that matches the video timing.

Technologies Used.
🔹 YOLOv8 – For real-time object detection in video frames.
🔹 OpenAI ChatGPT API – To filter sound-relevant objects and generate audio prompts.
🔹 Audio Generation Model (Stable Audio Model) – Produces realistic sound effects based on AI-generated prompts.
🔹 Python (NumPy, OpenCV, SoundFile, IPython) – Core libraries for video, audio, and AI processing.

How It Works (should in the future).
1️⃣ The system analyzes video frames and detects objects using YOLOv8.
2️⃣ The LLM (OpenAI ChatGPT API) filters out objects that don’t produce real-world sounds (e.g., traffic lights, umbrellas).
3️⃣ The filtered objects are sent to ChatGPT API again, which generates descriptive prompts for an Audio Model.
4️⃣ The Audio Model generates realistic sounds for each object.
5️⃣ The system calculates how long each object stays on screen and adjusts audio duration accordingly.
6️⃣ All small audio clips are merged into one final track, ensuring smooth synchronization with the video.
7️⃣ The final video is produced with an automatically generated soundscape!
