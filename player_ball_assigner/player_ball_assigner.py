import sys 
sys.path.append('../')
from utils import get_center, get_distance


class PlayerBallAssigner : 
    
    def __init__(self): 
        self.max_distance_with_ball = 70

    def assign_ball_to_player(self, player_track, ball_bbox): 
        
        ball_position = get_center(ball_bbox)

        minimum_distances = 99999
        assigned_player = -1 

        for player_id, player in player_track.items():
            player_bbox = player['bbox']

            distance_left = get_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = get_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_distance_with_ball : 
                if distance < minimum_distances : 
                    minimum_distances = distance
                    assigned_player = player_id

        return assigned_player

    