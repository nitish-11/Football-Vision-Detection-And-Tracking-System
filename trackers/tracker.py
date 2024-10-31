from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
# from utlis import get_center_of_bbox, get_bbox_width
from utlis import get_center_of_bbox, get_bbox_width

class Tracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions] #list of lists
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2']) 

        #Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [ {1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]  

        return ball_positions
    

    

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
            # break 
        return detections  # a list

    

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        
        tracks = {    #tracks dictionary created
            "players":[],
            "referees":[],
            "ball":[]
        }


        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # print(detection_supervision)
            # break

            # Convert Goalkeeper to player object

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':   #doubt1
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            
            # Tracks objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            # print(detection_with_tracks)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                


            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks,f)
        
        return tracks
    

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center,_ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame


    
    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    

    # def draw_team_ball_control(self,frame,frame_num,team_ball_control):
    #     # Draw a semi-transparent rectagle 
    #     overlay = frame.copy()
    #     cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
    #     alpha = 0.4
    #     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    #     team_ball_control_till_frame = team_ball_control[:frame_num+1]
    #     # Get the number of time each team had ball control
    #     team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
    #     team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
    #     team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
    #     team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

    #     cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
    #     cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

    #     return frame

    def identify_team_colors(self,tracks):
        team1_color, team2_color = None, None
        team_colors = {}

        # Traverse the dictionary of players in the first frame (frame 0)
        for player_id, player_data in tracks["players"][0].items():
            # Extract the team and color of the current player
            player_team = player_data["team"]
            player_color = player_data["team_color"]

            # Check if we have already assigned a color to this team
            if player_team == 1 and team1_color is None:
                team1_color = player_color
            elif player_team == 2 and team2_color is None:
                team2_color = player_color

            # If both team colors have been identified, stop the loop
            if team1_color is not None and team2_color is not None:
                break

        # Print and return the identified team colors
        print("Team 1 Color:", team1_color)
        print("Team 2 Color:", team2_color)
        return team1_color, team2_color



    
    # # def draw_team_ball_control(self, frame, frame_num, team_ball_control, tracks):
    #     # Draw a semi-transparent rectangle 
    #     overlay = frame.copy()
    #     cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    #     alpha = 0.4
    #     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


    #     # Extract ball control data up to the current frame
    #     team_ball_control_till_frame = team_ball_control[:frame_num+1]
        
    #     # Count frames controlled by each team, filtering for valid team IDs
    #     team_1_num_frames = (team_ball_control_till_frame == 1).sum()
    #     team_2_num_frames = (team_ball_control_till_frame == 2).sum()
        
    #     # Check to avoid division by zero
    #     total_frames = team_1_num_frames + team_2_num_frames
    #     if total_frames > 0:
    #         team_1 = team_1_num_frames / total_frames
    #         team_2 = team_2_num_frames / total_frames
    #     else:
    #         team_1, team_2 = 0, 0  # Default percentages if no control data is available

    #     # Display ball control percentages for each team
    #     cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    #     cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    #     return frame


    def draw_team_ball_control(self, frame, frame_num, team_ball_control, team1_color, team2_color):
        # Draw a semi-transparent rectangle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Extract ball control data up to the current frame
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]

        # Count frames controlled by each team, filtering for valid team IDs
        team_1_num_frames = (team_ball_control_till_frame == 1).sum()
        team_2_num_frames = (team_ball_control_till_frame == 2).sum()

        # Check to avoid division by zero
        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames > 0:
            team_1 = team_1_num_frames / total_frames
            team_2 = team_2_num_frames / total_frames
        else:
            team_1, team_2 = 0, 0  # Default percentages if no control data is available

        # Get team colors
        #team1_color, team2_color = self.identify_team_colors(tracks)

        # Draw color boxes for each team
        box_size = 40
        cv2.rectangle(frame, (1350, 870), (1350 + box_size, 870 + box_size), team1_color.astype(int).tolist(), -1)  # Team 1 color box
        cv2.rectangle(frame, (1350, 925), (1350 + box_size, 925 + box_size), team2_color.astype(int).tolist(), -1)  # Team 2 color box

        # Display ball control percentages for each team
        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        return frame



     
    def draw_annotations(self,video_frames, tracks,team_ball_control, team1_color, team2_color):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

                if player.get('has_ball',False):
                    frame = self.draw_triangle(frame, player["bbox"],(0,0,255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"],(0,165,255))
            
            # Draw ball 
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"],(0,255,0))


            # Draw Team Ball Control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control, team1_color, team2_color)

            output_video_frames.append(frame)

        return output_video_frames
    


    


   

                



