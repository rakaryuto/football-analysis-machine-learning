import cv2 
import pickle
import numpy as np
import sys 
import os
sys.path.append('../')
from utils import get_distance, get_xy_distance, get_center, get_foot_position

class CameraMovementEstimator(): 
    def __init__(self, frames): 

        self.minimum_distance = 5

        first_frames_grey = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frames_grey)
        mask_features[ : , 0:20 ] = 1
        mask_features[ : , 900:1050] = 1


        self.lk_params = dict(
            winSize = (15,15),
            maxLevel=2, 
            criteria= (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03)
        )
            
        self.features = dict(
            maxCorners = 100, 
            qualityLevel = 0.3, 
            minDistance = 3, 
            blockSize = 7, 
            mask = mask_features

        )

    def add_adjust_position_to_track(self, tracks, camera_movement_per_frame) : 
        for object, object_tracks in tracks.items() : 
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_position = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0] - camera_position[0], position[1] - camera_position[1])
                    # tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
    
        # Reading the stub from the file
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # movement for x and movement for y coordinates and multiplying by frames
        camera_movement = [[0, 0]]*len(frames)

        # extrcting old frames and features
        # Converting the first frame in frames to grayscale.
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # Detecting  good features to track in old_gray using the parameters defined in self.features.
        #  The **self.features syntax unpacks the dictionary into keyword arguments.
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params)

            # Measuring distance between new and old features
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            """
            Here we are iterating over the pairs of new and old features.

            new and old are points representing feature coordinates.
            .ravel() is a method in NumPy that flattens an array. It returns a contiguous flattened array, meaning it converts a multi-dimensional array into a one-dimensional array.

            new and old is likely a 2D array with the shape (1, 2) (or similar), containing the x and y coordinates of a feature point.
            new.ravel() flattens this 2D array into a 1D array with the shape (2,), which means new_features_point will be a 1D array containing the x and y coordinates
            """
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # calculating the distance between the new and old features using a custom function measure_distance.
                distance = get_distance(
                    new_features_point, old_features_point)

                # If the current distance is greater than max_distance, update max_distance and calculate the x and y movement using a custom function measure_xy_distance.
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = get_xy_distance(
                        old_features_point, new_features_point)

            """
            Here we check the max_distance (the maximum distance between new and old features calculated in the loop) is greater than a threshold value, self.minimum_distance.
            If the condition is true we update the camera_movement list at the index frame_num (the current frame number).
            camera_movement_x and camera_movement_y are the x and y components of the movement vector, calculated as the largest movement of features between the two frames.

            After recording the camera movement, this line detects new features in the current frame (now stored in frame_gray).
            cv2.goodFeaturesToTrack is an OpenCV function used to detect good features (corners) to track in an image.
            The **self.features syntax unpacks the dictionary self.features into keyword arguments for the cv2.goodFeaturesToTrack function. This dictionary contains parameters like maxCorners, qualityLevel, minDistance, blockSize, and mask.
            """
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [
                    camera_movement_x, camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features)

            # Updating old_gray to be the current frame's grayscale image.
            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame): 
        
        output_frames = []
        for frame_num, frame in enumerate(frames): 
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"Camera Movement X: {x_movement:.2f}", (
                10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            frame = cv2.putText(frame, f"Camera Movement Y: {y_movement:.2f}", (
                10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            output_frames.append(frame)

        return output_frames