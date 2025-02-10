# YOLO Object Counting Project

This project implements object counting using the YOLOv11 model. It processes video input to count specified objects within defined regions and outputs the results to a new video file.

## Project Structure

```
yolo-object-counting
├── input_videos/         # Folder for input videos
├── output_videos/        # Folder for output videos
├── src
│   ├── main.py           # Main entry point for video processing and object counting
│   └── utils.py          # Utility functions for model loading, frame processing, and video conversion
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone https://github.com/yourusername/yolo-object-counting.git
   cd yolo-object-counting
   ```

2. **Install dependencies:**
   Ensure you have Python installed, then run:
   ```
   pip install -r requirements.txt
   ```

3. **Install FFmpeg:**
   Follow the instructions to install FFmpeg on your system: https://ffmpeg.org/download.html

## Usage

1. Place your video file in the `input_videos` folder.
2. Modify the `src/main.py` file to specify the path to your video file and the desired counting region.
3. Run the application:
   ```
   python src/main.py
   ```

4. The output video will be saved as `object_counting_output.mp4` in the `output_videos` folder.

## Notes

- Ensure you have the YOLOv11 model weights (`yolo11n.pt`) available in the project directory or specify the correct path in the code.
- Adjust the region points in `main.py` to fit your counting needs.

## License

This project is licensed under the MIT License.