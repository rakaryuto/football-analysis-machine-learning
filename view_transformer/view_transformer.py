import numpy as np 
import cv2 

class ViewTransformer(): 

    def __init__(self): 
        court_width = 68
        court_length = 23.32

        self.pixels_vertices = np.array([
            [110,1035],
            [265, 275], 
            [910, 260],
            [1640, 915]
        ])

        self.target_vertices = np.array([
            [0, court_width],
            [0,0], 
            [court_length, 0],
            [court_length, court_width] 
        ])

        self.pixels_vertices = self.pixels_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformers = cv2.getPerspectiveTransform(self.pixels_vertices, self.target_vertices)

    def transformed_point(self, point) : 
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixels_vertices, p, False) >= 0 
        if not is_inside : 
            return None 
        
        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformers)

        return transformed_point.reshape(-1, 2)

    def add_transform_position_to_track(self, tracks) : 
        for object_id, object_track in tracks.items(): 
            for frame_num, track in enumerate(object_track) : 
                for track_id, track_info in track.items(): 
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    transformed_position = self.transformed_point(position)
                    if transformed_position is not None : 
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object_id][frame_num][track_id]['transformed_position'] = transformed_position
                
