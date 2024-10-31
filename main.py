from utlis import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
import cv2



def main():
    #Read Video
    video_frames = read_video('input_videos/input_testing/in_video18.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best_2.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path = 'stubs/track_stubs18.pkl')
    


    # # Cropped image 
    # for track_id, player in tracks['players'][0].items():
    #     bbox = player['bbox']
    #     frame = video_frames[0]

    #     #crop the bbox from frame
    #     cropped_image = frame[int(bbox[1]): int(bbox[3]),
    #                           int(bbox[0]): int(bbox[2])]
        
    #     # saved cropped image
    #     cv2.imwrite(f'output_videos/cropped_image_1.jpg', cropped_image)
    #     break


    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
    



    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    # Define a default team ID for the first frame if no ball possession is found
    default_team_id = 1

    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # For the first frame, assign it to the default team if no previous data exists
            if frame_num == 0:
                team_ball_control.append(default_team_id)
            else:
                # For subsequent frames, use the last team's value
                team_ball_control.append(team_ball_control[-1])

    # Convert the list to a numpy array after the loop
    team_ball_control = np.array(team_ball_control)

            
    # Get team colors
    team1_color, team2_color = tracker.identify_team_colors(tracks)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control, team1_color, team2_color)


    # Saved video
    save_video(output_video_frames, 'output_videos/testing/out_video18_t3.avi')


if __name__ == '__main__':
    main()


