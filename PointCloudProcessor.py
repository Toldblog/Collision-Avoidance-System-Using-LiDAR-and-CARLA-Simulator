class PointCloudProcessor:
    def __init__(self, output_dir=None, store_frames=True, eps=EPS, min_samples=MIN_SAMPLES):
        self.eps = eps
        self.min_samples = min_samples
        self.output_dir = output_dir
        self.store_frames = store_frames

        self.frame_count = 0

        self.previous_centroids = []
        self.previous_cluster_ids = []

        # Initialize tracking history and IDs
        self.cluster_history = []
        self.tracking_id = 0
        self.cluster_color_map = {}
        self.previous_centroids_map = {}
        self.previous_cluster_points_map = {}

    def process_frame(self, current_frame):
        # print(f"Frame: {self.frame_count}--------------------------------")
        road_points, road_height = filter_road(current_frame)
        x_boundary_map, y_boundary_map = build_road_boundary_map(road_points)
        vehicle_points = []
        for _, point in enumerate(current_frame):
            if is_point_on_road(x_boundary_map, y_boundary_map, point, road_height):
                vehicle_points.append(point)

        vehicle_points = np.array(vehicle_points)

        road_points, road_height = filter_road(current_frame)
        x_boundary_map, y_boundary_map = build_road_boundary_map(road_points)

        clustering = DBSCAN(
            eps=self.eps, min_samples=self.min_samples).fit(vehicle_points)
        labels = clustering.labels_

        current_clusters_points = []
        current_centroids = []
        movement_vectors = []
        current_bboxs = []

        for cluster_label in set(labels):
            if cluster_label == -1:
                continue

            cluster_points = vehicle_points[labels == cluster_label]

            if np.any(cluster_points[:, 2] >= 1):
                continue

            if np.any(cluster_points[:, 2] >= -1):
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                point = np.asarray(bbox.get_box_points())
                d_height = np.linalg.norm(point[0] - point[3])
                d1 = np.linalg.norm(point[0] - point[2])
                if d_height > 1 and d1 > 1:
                    current_clusters_points.append(cluster_points)
                    current_centroids.append(bbox.get_center())
                    current_bboxs.append(bbox)

        current_centroids = np.array(current_centroids)

        if self.frame_count == 0:
            cluster_ids = np.arange(len(current_centroids))
            self.tracking_id = len(current_centroids)
        else:
            cluster_ids, self.tracking_id = self.track_clusters(
                current_centroids, self.previous_centroids, self.previous_cluster_ids)

        for i, cluster in enumerate(current_clusters_points):
            if cluster_ids[i] in self.previous_cluster_points_map:
                previous_cluster_points = self.previous_cluster_points_map[cluster_ids[i]]
                movement_result = find_cluster_movement(
                    cluster[:, :2], previous_cluster_points[:, :2])
                if movement_result is not None:
                    h_i, t_i = movement_result
                    movement_vectors.append((h_i, t_i))

            self.previous_cluster_points_map[cluster_ids[i]] = cluster

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        frame_filename = os.path.join(self.output_dir, f"frame_{
                                      self.frame_count:04d}.png")

        self.previous_centroids = current_centroids
        self.previous_cluster_ids = cluster_ids
        self.frame_count += 1
        # print("Process 0")
        if current_clusters_points:
            print(f"Frame: {self.frame_count}")
            vehicle_points_2d = np.vstack(current_clusters_points)
            cropped_image = self.visualize_2d_points(
                vehicle_points_2d, x_boundary_map, movement_vectors, current_bboxs, frame_filename)
            return cropped_image
        else:
            return None

    def track_clusters(self, current_centroids, previous_centroids, prev_cluster_ids, threshold=DIST_THRESHOLD_TRACKING):
        if len(previous_centroids) == 0:
            return np.arange(len(current_centroids)) + self.tracking_id, self.tracking_id + len(current_centroids)

        if len(current_centroids) == 0:
            return np.array([]), self.tracking_id

        tree = KDTree(previous_centroids)
        assigned_ids = np.full(len(current_centroids), -1)
        distances, indices = tree.query(current_centroids)

        # Assign tracking IDs based on distance threshold
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if dist < threshold:
                assigned_ids[i] = prev_cluster_ids[idx]

        new_clusters = (assigned_ids == -1)
        assigned_ids[new_clusters] = np.arange(
            self.tracking_id, self.tracking_id + np.sum(new_clusters))

        return assigned_ids, self.tracking_id + np.sum(new_clusters)

    def visualize_2d_points(self, points, x_boundary_map, movement_vectors, current_bboxs, save_path):
        fig, ax = plt.subplots(figsize=(6, 6))
        canvas = FigureCanvas(fig)

        # Extract and sort x and y boundaries for the road boundary fill
        y_values = sorted(x_boundary_map.keys())
        x_min_values = [x_boundary_map[y]['min'] for y in y_values]
        x_max_values = [x_boundary_map[y]['max'] for y in y_values]

        y_values_flipped = [-y for y in y_values]

        # Plot the road area
        ax.fill_betweenx(y_values_flipped, x_min_values,
                         x_max_values, color='white', label='Road Area')
        ax.fill_betweenx(y_values_flipped, min(x_min_values) - 10,
                         x_min_values, color='black', label='Outside Road Area (Left)')
        ax.fill_betweenx(y_values_flipped, x_max_values, max(
            x_max_values) + 10, color='black', label='Outside Road Area (Right)')

        # Plot a fixed teal bounding box
        teal_bbox_x = [-1.1, 1.1, 1.1, -1.1, -1.1]  # X-coordinates
        teal_bbox_y = [-2.4, -2.4, 2.4, 2.4, -2.4]  # Y-coordinates
        # ax.plot(teal_bbox_x, teal_bbox_y, color='blue', lw=2, label='Fixed Teal BBox')

        # Plot the points
        points = np.asarray(points)
        if points.ndim == 1:
            # print("Points array is 1D, reshaping to 2D.")
            points = points.reshape(-1, 2)

        # Apply DBSCAN clustering
        x_points = -points[:, 0]
        y_points = points[:, 1]
        x_points, y_points = y_points, x_points  # Swap axes

        points[:, 0], points[:, 1] = points[:, 1].copy(), -points[:, 0].copy()

        ax.scatter(points[:, 0], points[:, 1],
                   c='black', s=7, label='h_i Points')

        for h_i, t_i in movement_vectors:
            # Plot movement vector as an arrow
            mean_h_i = np.mean(h_i, axis=0)
            # mean_h_i = mean_h_i*2
            mean_t_i = np.mean(t_i, axis=0)
            start_x, start_y = mean_t_i[1],  -mean_t_i[0]
            end_x, end_y = mean_h_i[1],  -mean_h_i[0]
            dx = end_x - start_x
            dy = end_y - start_y
            ax.arrow(start_x, start_y, dx*2.5, dy*2.5, width=3,
                     head_width=0.0, head_length=0.0, fc='black', ec='black')

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.axis('equal')  # Ensure equal scaling
        ax.set_xlim(-30, 30)  # Adjust limits as needed
        ax.set_ylim(-30, 30)
        ax.set_aspect('equal', adjustable='box')
        # plt.show()

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(int(height), int(width), 4)[:, :, :3]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(
            gray_image, 127, 255, cv2.THRESH_BINARY)
        cropped_image = crop_center_rectangle(binary_image)

        return cropped_image
