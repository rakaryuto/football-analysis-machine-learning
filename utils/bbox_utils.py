def get_center(bbox):
    x_center = (bbox[0] + bbox[2]) / 2
    y_center = (bbox[1] + bbox[3]) / 2
    return (int(x_center), int(y_center))

def get_width(bbox): 
    return bbox[2]-bbox[0]

def get_distance(p1, p2): 
    distance = ((p1[0]-p2[0])**2 +  (p1[1]-p2[1])**2)**0.5
    return distance

def get_xy_distance(p1, p2): 
    distance_x = p1[0] - p2[0]
    distance_y = p1[1] - p2[1]
    return distance_x, distance_y

def get_foot_position(bbox) : 
    x1, y1, x2, y2 = bbox
    return int((x1+x2) / 2), int(y2)
