import cv2
from ultralytics import solutions
from utils import convert_video_format
from utils import process_frame
from utils import release_resources
from utils import convert_video_format
from utils import draw_custom_counter


def main():
    # Define input and output paths for video processing. Adjust the input video name and path as needed.
    # Input: High resolution video (3840x2160) at 25fps in this example
    # Output: Temporary AVI format for processing, final output in MP4
    input_video_path = "../input_videos/birds_3840_2160_25fps.mp4"
    output_video_path = "../output_videos/object_counting_output.avi"
    final_output_video_path = "../output_videos/object_counting_output.mp4"

    # Initialize video capture and extract video properties
    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define counting regions
    # Two types of regions are supported:
    # 1. Rectangle region: Define 4 points for a rectangle area
    # 2. Line counting: Define 2 points for a vertical/horizontal line
    # region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # For rectangle region counting
    region_points = [(2300, 20), (2300, 2100)]  # For line counting - vertical line at x=2300

    # Initialize video writer with same properties as input video
    # Using 'mp4v' codec for temporary AVI output
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize YOLOv11 object counter
    # - show: Enable visualization
    # - region: Specify counting region/line
    # - model: Use YOLOv11-nano model
    # - classes: You can define class(es) in as per the COCO dataset
    # - show_in/show_out: Display counts for objects entering/leaving the region
    counter = solutions.ObjectCounter(
        show=True,
        region=region_points,
        model="yolo11n.pt",
        classes=[14],  # Class 14 represents birds in COCO dataset
        show_in=True,  # Display count of objects entering the region
        show_out=True,  # Display count of objects leaving the region    
        )

    # Main video processing loop
    while cap.isOpened():
        # Read frame from video
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # Process frame through YOLO model and get object detections
        im0 = process_frame(im0, counter)

        # Retrieve current object counts
        in_count = counter.in_count   # Objects that entered the region
        out_count = counter.out_count # Objects that left the region

        # Add custom visualization of counter statistics
        im0 = draw_custom_counter(im0, in_count, out_count)

        # Write processed frame to output video
        video_writer.write(im0)

    # Clean up resources
    release_resources(cap, video_writer)

    # Convert final output from AVI to MP4 format for better compatibility
    convert_video_format(output_video_path, final_output_video_path)

if __name__ == "__main__":
    main()