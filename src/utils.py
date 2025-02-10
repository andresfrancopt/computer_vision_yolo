import subprocess
import cv2

def load_model(model_path):
    """
    Load the YOLO model from the specified path
    Args:
        model_path: Path to the YOLO model weights file
    Returns:
        YOLO model instance
    """
    from ultralytics import YOLO
    return YOLO(model_path)

def process_frame(frame, counter):
    """
    Process a single video frame through the object counter
    Args:
        frame: Input video frame (numpy array)
        counter: ObjectCounter instance from ultralytics.solutions
    Returns:
        Processed frame with detection visualizations
    """
    return counter.count(frame)

def initialize_video_capture(video_path):
    """
    Initialize OpenCV video capture for the input video file
    Args:
        video_path: Path to the input video file
    Returns:
        cv2.VideoCapture object
    Raises:
        AssertionError if video file cannot be opened
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    return cap

def release_resources(cap, video_writer):
    """
    Clean up video capture and writer resources
    Args:
        cap: OpenCV VideoCapture object
        video_writer: OpenCV VideoWriter object
    """
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def convert_video_format(input_path, output_path):
    """
    Convert video from AVI to MP4 format using FFmpeg
    Args:
        input_path: Path to input AVI video
        output_path: Path for output MP4 video
    Settings:
        - libx264: H.264 video codec
        - crf 18: High quality compression
        - preset slow: Better compression at cost of encoding speed
    """
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vcodec', 'libx264',
        '-crf', '18',
        '-preset', 'slow',
        output_path
    ]
    subprocess.run(command, check=True)

def draw_custom_counter(im0, in_count, out_count):
    """
    Draw object counting statistics overlay on the video frame
    Args:
        im0: Input frame (numpy array)
        in_count: Number of objects that entered
        out_count: Number of objects that left
    Returns:
        Frame with counter overlay
    
    Note: Counter box is positioned at (3540, 20) with size 230x120 pixels,
          designed for 3840x2160 resolution videos
    """
    # Define the position and size of the counter box
    box_x, box_y, box_w, box_h = 3540, 20, 230, 120

    # Draw the background rectangle
    cv2.rectangle(im0, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)

    # Define the text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.3
    font_color = (255, 255, 255)  # White color in BGR
    thickness = 3

    # Draw the in count text
    in_text = f"In: {in_count}"
    cv2.putText(im0, in_text, (box_x + 20, box_y + 40), font, font_scale, font_color, thickness)

    # Draw the out count text
    out_text = f"Out: {out_count}"
    cv2.putText(im0, out_text, (box_x + 20, box_y + 100), font, font_scale, font_color, thickness)

    return im0