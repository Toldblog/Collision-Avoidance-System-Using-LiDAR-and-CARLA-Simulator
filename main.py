import numpy as np

from PointCloudProcessor import *

def main():
    try:
        cloud_point_data = np.load(
            'cloud_point_data4k.pkl', allow_pickle=True)

        output_dir = 'frames'
        output_video = 'output_videos/white_black_v2.mov'

        processor = PointCloudProcessor(output_dir)
        for frame_idx in range(0, len(cloud_point_data), 1):  # Increment by 5
            cropped_image = processor.process_frame(
                cloud_point_data[frame_idx])

        # processor.close_vis()

        # create_video_from_frames(output_dir, output_video)

    finally:
        print("Success")


if __name__ == "__main__":
    main()