class SpeedandDistanceEstimator(): 
    def __init__(self): 
        
        self.frame_window = 5
        self.frame_rate = 24 

    def add_speed_and_distance(self, tracks):
        for object, object_track in tracks.items(): 
            if object == 'ball' or object == 'refere' : 
                continue 
            
 