from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
import numpy as np
import cv2

def main() : 

    #Read Video 
    video_frames = read_video("./input-videos/08fd33_4.mp4")

    #Insert object tracking
    tracker = Tracker('./models/best.pt')
    #List of track id
    tracks = tracker.get_object_track(video_frames, read_from_stub=True, stub_path="./stubs/track_stubs.pkl")

    #Add position to Tracker 
    tracker.add_position_to_tracks(tracks)

    #Camera Movement Estimator 
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='./stubs/camera_movement_stubs.pkl')

    camera_movement_estimator.add_adjust_position_to_track(tracks, camera_movement_per_frame)

    #Transform View into flat board
    view_transformer = ViewTransformer()
    view_transformer.add_transform_position_to_track(tracks)

    #Interpolate missing information from ball position
    tracks['ball'] = tracker.interpolate_ball_position(tracks["ball"])

    #Assign team color
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num , player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items(): 
            team = team_assigner.get_player_team(video_frames[frame_num], 
                                                track['bbox'], 
                                                player_id
                                                )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Assigning player with ball 
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player in enumerate(tracks['players']): 
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player, ball_bbox)

        if assigned_player != -1 : 
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else : 
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # #cropped image for analyzing
    # for track_id, player in tracks["players"][0].items():
    #     bbox = player["bbox"]
    #     frame = video_frames[0]

    #     cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    #     cv2.imwrite("./output_videos/cropped_img.jpg", cropped_img)
    #     break 

    #Draw an annotation above video frames
    output_video_frames = tracker.draw_annotation(video_frames, tracks, team_ball_control)

    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # print("============ Saving Videos ===========")
    # save_video(video_frames)
    # save_video(output_video_frames, 'tracked_output.mp4')
    save_video(output_video_frames, 'movement_output.mp4')

if __name__ == "__main__" :
    main()