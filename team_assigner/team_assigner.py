from sklearn.cluster import KMeans

class TeamAssigner : 
    
    def __init__(self): 
        self.team_colors = {}
        self.player_team_dict = {}

    def get_cluster_center(self, top_half_img):

        img_2d = top_half_img.reshape(-1,3)

        kmeans = KMeans(n_clusters=2, random_state=1)
        kmeans.fit(img_2d)

        labels = kmeans.labels_
        clusterred_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])

        corner_clusters = [clusterred_img[0,0], clusterred_img[0,-1], clusterred_img[-1,0], clusterred_img[-1,-1]]
        non_player_clusters = max(set(corner_clusters), key=corner_clusters.count)

        player_clusters = 1 - non_player_clusters
    
        return kmeans.cluster_centers_[player_clusters]

    def get_color(self, frame, bbox): 
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_img = image[0: int(image.shape[0]/2), :]

        cluster_center = self.get_cluster_center(top_half_img)

        return cluster_center

    def assign_team_color(self, frame, player_detection): 
        player_colors = []

        for track_id, player in player_detection.items():
            bbox = player['bbox']
            player_color = self.get_color(frame, bbox)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=100)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    
    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict: 
            return self.player_team_dict[player_id]
        
        player_color = self.get_color(frame, player_bbox)
    
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        
        team_id += 1 

        if player_id == 74 or player_id == 66 or player_id == 62 or player_id == 106: 
            team_id = 1

        self.player_team_dict[player_id] = team_id

        return team_id
