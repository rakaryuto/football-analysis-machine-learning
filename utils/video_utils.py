import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return frames
    
    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames

def save_video(output_video_frames, file_name):
    if not output_video_frames:
        print("Error: No frames to save.")
        return
    
    fourcc = cv2.VideoWriter.fourcc(*"MJPG")
    print("FOURCC: ")
    print(fourcc)
    
    if output_video_frames:
        height, width = output_video_frames[0].shape[:2]
    else:
        raise ValueError("No frames to save in output_video_frames.")   
    
    # Ensure the output directory exists
    output_dir = 'output_videos/'
    os.makedirs(output_dir, exist_ok=True)
    
    out = cv2.VideoWriter(os.path.join(output_dir, file_name), fourcc, 24, (width, height))
    print(f'Video_Writer : {out}')

    for frame in output_video_frames:
        out.write(frame)
        
    out.release()
    print(f"Video saved to {output_dir}output.mp4")

