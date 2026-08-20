import cv2


class VideoSampler:
    """
    Samples frames from a video at a target FPS.

    Frames are returned together with their original frame index
    and timestamp.
    """

    def __init__(self, video_path, target_fps=10):
        self.video_path = video_path
        self.target_fps = target_fps

    def __iter__(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)

        if original_fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS for video: {self.video_path}")

        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            timestamp = frame_index / original_fps

            # Sample based on time rather than assuming a fixed FPS.
            sample_index = int(timestamp * self.target_fps)

            next_timestamp = ((frame_index + 1) / original_fps)

            next_sample_index = int(next_timestamp * self.target_fps)

            if sample_index != next_sample_index:
                yield frame_index, timestamp, frame

            frame_index += 1

        cap.release()