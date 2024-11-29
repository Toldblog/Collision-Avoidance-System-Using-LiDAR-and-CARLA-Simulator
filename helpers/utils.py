import os
import cv2
import numpy as np
import open3d as o3d
import pickle
from scipy.interpolate import interp1d
from collections import defaultdict
import matplotlib.pyplot as plt 

from constants import *


def create_video_from_frames(frames_path, output_video, frame_rate=FRAME_RATE):
    """Combines saved frame images into a video."""
    # Get the list of all frame files
    frame_files = sorted([f for f in os.listdir(
        frames_path) if f.startswith('frame_') and f.endswith('.png')])

    if not frame_files:
        print("Error: No frame images found.")
        return

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
    if first_frame is None:
        print("Error: Could not load the first frame.")
        return

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(
        output_video, fourcc, frame_rate, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_path, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video_writer.write(frame)
        else:
            print(f"Error: Could not load frame {frame_file}")

    video_writer.release()
    print(f"Video saved as {output_video}")


def find_velocity_kmh(start, end, time=1/FRAME_RATE):
    """Computes velocity in km/h between two points."""
    v = np.linalg.norm(end - start) / time
    kmh = int(3.6 * v)
    return kmh


def create_arrows(start, end, bbox, previous_direction=None, scale_factor=ARROW_SCALE_FACTOR):
    kmh = find_velocity_kmh(end, start)
    if kmh > VELOCITY_THRESHOLD:
        direction = end - start
        direction[2] = 0

        if previous_direction is not None:
            updated_direction = 0.9 * previous_direction + 0.1 * direction  # Apply EMA
        else:
            updated_direction = direction
        length = np.linalg.norm(updated_direction)

        if length == 0:
            return None

        bbox_corners = np.asarray(bbox.get_box_points())
        bottom_corners = bbox_corners[bbox_corners[:, 2] == np.min(
            bbox_corners[:, 2])]
        lines = []

        # Generate a line from each top corner in the specified direction
        for corner in bottom_corners:
            end_point = corner + length * direction * 40
            line_points = np.array([corner, end_point])
            line = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(line_points),
                lines=o3d.utility.Vector2iVector([[0, 1]])
            )
            line.paint_uniform_color([1, 0, 1])  # Set line color to purple
            lines.append(line)
        return lines

    return None


def find_top_edge_centers(bbox):
    bbox_corners = np.asarray(bbox.get_box_points())
    top_face_points = bbox_corners[bbox_corners[:, 2]
                                   == np.min(bbox_corners[:, 2])]
    edges = {
        ('0', '1'): np.linalg.norm(top_face_points[0] - top_face_points[1]),
        ('0', '2'): np.linalg.norm(top_face_points[0] - top_face_points[2]),
        ('0', '3'): np.linalg.norm(top_face_points[0] - top_face_points[3]),
        ('1', '2'): np.linalg.norm(top_face_points[1] - top_face_points[2]),
        ('1', '3'): np.linalg.norm(top_face_points[1] - top_face_points[3]),
        ('2', '3'): np.linalg.norm(top_face_points[2] - top_face_points[3])
    }

    sorted_edges = sorted(edges.items(), key=lambda x: x[1], reverse=True)
    a, b = sorted_edges[4][0]
    c, d = sorted_edges[5][0]
    front = (top_face_points[int(a)]+top_face_points[int(b)])/2
    front[2] = top_face_points[int(a)][2]
    back = (top_face_points[int(c)]+top_face_points[int(d)])/2
    back[2] = top_face_points[int(c)][2]
    return front, back


def create_area_between_arrows(lines, color):
    """Create a quadrilateral mesh connecting the ends of the four lines."""
    # Get end points of each line (each line has points at index 0 and 1)
    end_points = [np.asarray(line.points)[1] for line in lines]

    # Form a quadrilateral by connecting the end points
    vertices = np.array(end_points)
    triangles = np.array([[0, 1, 2], [2, 3, 0]])

    # Create the area mesh
    area_mesh = o3d.geometry.TriangleMesh()
    area_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    area_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    area_mesh.paint_uniform_color(color)

    return area_mesh


def merge_clusters(clusters):
    """Merge two clusters into one."""
    merged_cluster = np.vstack(clusters)  # Stack them together
    return merged_cluster


def calculate_car_direction(bbox):
    """Calculates the direction of the car using the front and back top edges of the bounding box."""
    front_top_center, back_top_center = find_top_edge_centers(bbox)
    # Calculate the car's forward direction (from back to front of the car)
    car_direction = front_top_center - back_top_center
    car_direction[2] = 0  # Keep the movement in the X-Y plane
    norm_direction = car_direction / np.linalg.norm(car_direction)
    return norm_direction


def maintain_speed(s):
    """Function to maintain the desired speed."""
    if s >= PREFERRED_SPEED:
        return 0
    elif s < PREFERRED_SPEED - SPEED_THRESHOLD:
        return 0.9  # High throttle
    else:
        return 0.4  # Moderate throttle


def save_point_cloud_data(data, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Save cloud point data succesfully at {output_path}")


def load_point_cloud_data(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    print("Load cloud point data succesfully")
    print(f"Cloud point data's length: {len(data)}")
    print(f"Cloud point data[0]'s shape: {data[0].shape}")
    return data


def interpolate_3d_points_ignore_z(points, z=0):
    x_values = points[:, 0]
    y_values = points[:, 1]
    f = interp1d(x_values, y_values, kind='linear')
    x_new = np.linspace(min(x_values), max(x_values), num=10000)
    y_new = f(x_new)
    new_points = np.column_stack((x_new, y_new))
    z_new = np.full(new_points.shape[0], z)
    new_points_with_z = np.column_stack((new_points, z_new))
    points = np.vstack((points, new_points_with_z))
    return points


def generate_3d_circle_points(radius, height, num_points=1000):
    r = radius * np.sqrt(np.random.rand(num_points))
    theta = np.random.rand(num_points) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full(num_points, height)  # z = 0 for a circle in the xy-plane
    circle_points = np.vstack((x, y, z)).T
    return circle_points


def generate_3d_rectangle_points(a, b, height, num_points=5000):
    x = np.random.uniform(-3*a/10, 7*a/10, num_points)
    y = np.random.uniform(-b/2, b/2, num_points)
    z = np.full(num_points, height)
    rectangle_points = np.vstack((x, y, z)).T
    return rectangle_points


def filter_road(pcd):
    z_values = [point[2] for point in pcd]
    unique_z_values = sorted(set(z_values))
    # average_height = np.sum(unique_z_values[:100]) / 100
    # road_height = np.ceil(average_height * 10) / 10
    # height_filter = pcd[:, 2] <= road_height
    # combined_points = pcd[height_filter]

    road_height = np.ceil(unique_z_values[0] * 10) / 10
    height_tolerance = 0.05  # Define a tolerance level for "closeness"
    height_filter = np.isclose(pcd[:, 2], road_height, atol=height_tolerance)
    combined_points = pcd[height_filter]

    circle_points = generate_3d_circle_points(
        radius=4, height=road_height, num_points=500)
    rectangle_points = generate_3d_rectangle_points(
        a=15, b=5, height=road_height, num_points=500)
    combined_points = np.vstack(
        (combined_points, circle_points, rectangle_points))
    return combined_points, road_height


def build_road_boundary_map(road_points):
    # Initialize the boundary maps
    x_boundary_map = defaultdict(
        lambda: {'min': float('inf'), 'max': -float('inf')})
    y_boundary_map = defaultdict(
        lambda: {'min': float('inf'), 'max': -float('inf')})

    # Populate the maps with min/max x and y values for each y and x, respectively
    for point in road_points:
        x, y, _ = np.round(point)  # We ignore the z value

        x_boundary_map[x]['min'] = min(x_boundary_map[x]['min'], y)
        x_boundary_map[x]['max'] = max(x_boundary_map[x]['max'], y)

        # Update y map with the min/max x values
        y_boundary_map[y]['min'] = min(y_boundary_map[y]['min'], x)
        y_boundary_map[y]['max'] = max(y_boundary_map[y]['max'], x)

    return x_boundary_map, y_boundary_map


def is_point_on_road(x_boundary_map, y_boundary_map, point, road_height):
    x, y, z = point
    x_r, y_r, _ = np.round(point)
    _, _, z_r = np.floor(point * 10) / 10
    if ((x_boundary_map[x_r]['min'] <= y <= x_boundary_map[x_r]['max']) and (y_boundary_map[y_r]['min'] <= x <= y_boundary_map[y_r]['max']) and z_r > road_height):
        return True

    return False


def find_vehicle_on_road_per_frame(frames):
    road_points, road_height = filter_road(frames)
    x_boundary_map, y_boundary_map = build_road_boundary_map(road_points)
    vehicle_points = []
    for i, point in enumerate(frames):
        if is_point_on_road(x_boundary_map, y_boundary_map, point, road_height):
            vehicle_points.append(point)
    return vehicle_points


def visualize_highest_points(pcd, bounding_boxes, arrows, save_path=None):

    points = np.asarray(pcd.points)
    x_points = -points[:, 0]
    y_points = points[:, 1]
    x_points, y_points = y_points, x_points
    plt.figure(figsize=(6, 6))
    plt.scatter(x_points, y_points, c='blue', s=1, label='Point Cloud')

    if len(arrows) > 0:
        arrows = np.asarray(arrows)

        for arrow in arrows:
            arrow = np.asarray(arrow)
            arrow = rearrange_arrow(arrow)

            previous_line = arrow[-1]
            for line in arrow:
                line_points = line
                # line_points[1], line_points[2] = line_points[2], line_points[1]
                x_arrow = -line_points[:, 0]
                y_arrow = line_points[:, 1]
                x_arrow, y_arrow = y_arrow, x_arrow
                plt.plot(x_arrow, y_arrow, color='red', lw=2, label='Arrow')

                x_polygon, y_polygon = extract_polygon(line, previous_line)
                plt.fill(y_polygon, -x_polygon, color='red', alpha=0.3)
                previous_line = line

    # for bbox in bounding_boxes:
    #     bbox_points = np.asarray(bbox.get_box_points())
    #     x_bbox = -bbox_points[[0, 1, 7, 2, 0], 0]
    #     y_bbox = bbox_points[[0, 1, 7, 2, 0], 1]
    #     x_bbox, y_bbox = y_bbox, x_bbox
    #     plt.plot(x_bbox, y_bbox, color='red', lw=2, label='Bounding Box')

    # Draw the new 2D projected bounding boxes
    # for bbox_2d in movement_direction_areas:
    #     x_bbox_2d = -bbox_2d[:, 0]
    #     y_bbox_2d = bbox_2d[:, 1]
    #     x_bbox = np.append(x_bbox_2d, x_bbox_2d[0])
    #     y_bbox = np.append(y_bbox_2d, y_bbox_2d[0])
    #     x_bbox[2], x_bbox[3] = x_bbox[3], x_bbox[2]
    #     y_bbox[2], y_bbox[3] = y_bbox[3], y_bbox[2]

    #     plt.plot(y_bbox, x_bbox, color='red', lw=2, label='Projected BBox')

    # X-coordinates (in swapped axes)
    teal_bbox_x = [-1.1, 1.1, 1.1, -1.1, -1.1]
    # Y-coordinates (in swapped axes)
    teal_bbox_y = [-2.4, -2.4, 2.4, 2.4, -2.4]
    plt.plot(teal_bbox_x, teal_bbox_y, color='teal',
             lw=2, label='Fixed Teal BBox')

    plt.xlim(-30, 30)  # 40 units width, centered at 0
    plt.ylim(-30, 30)

    plt.gca().set_aspect('equal', adjustable='box')  # Keep the aspect ratio 1:1

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    # plt.show()
    plt.close()


def find_cluster_movement(cluster_i, cluster_i_minus_1, threshold=0.2):
    h_i, t_i = [], []
    for point in cluster_i:
        distances = np.linalg.norm(cluster_i_minus_1 - point, axis=1)
        if np.all(distances > threshold):
            # if point not in cluster_i_minus_1:
            h_i.append(point)

    for point in cluster_i_minus_1:
        distances = np.linalg.norm(cluster_i - point, axis=1)
        if np.all(distances > threshold):
            # if point not in cluster_i:
            t_i.append(point)

    # Convert to numpy arrays
    h_i = np.array(h_i)
    t_i = np.array(t_i)
    # print(len(h_i), len(t_i))

    # Step 4: Calculate mean points of h_i and t_i
    if len(h_i) == 0 or len(t_i) == 0:
        return None

    return h_i, t_i


def extract_polygon(line1, line2):
    x_polygon = np.array([line1[0, 0], line1[1, 0], line2[1, 0], line2[0, 0]])
    y_polygon = np.array([line1[0, 1], line1[1, 1], line2[1, 1], line2[0, 1]])
    return x_polygon, y_polygon


def rearrange_arrow(arrow):
    arrow = np.array(arrow)
    rearranged_arrow = [arrow[0]]
    remaining_lines = list(arrow[1:])

    while remaining_lines:
        for i, line in enumerate(remaining_lines):
            if line[0][0] == rearranged_arrow[-1][0][0] or line[0][1] == rearranged_arrow[-1][0][1]:
                rearranged_arrow.append(line)
                remaining_lines.pop(i)
                break
        else:
            # Fallback in case no match found
            rearranged_arrow.append(remaining_lines.pop(0))

    return rearranged_arrow


def find_closest_distances(cluster_points, num_closest=5):
    if len(cluster_points) < 2:
        return []  # Not enough points to calculate distances

    # Compute pairwise distances
    dist_matrix = distance.cdist(
        cluster_points, cluster_points, metric='euclidean')

    # Set diagonal to infinity to ignore zero distances (distance to itself)
    np.fill_diagonal(dist_matrix, np.inf)

    # Flatten the distance matrix and find the smallest distances
    closest_distances = np.sort(dist_matrix.ravel())[:num_closest]

    return closest_distances


def crop_center_rectangle(image, target_height=440, target_width=440):
    h, w, c = image.shape  # Get the shape of the original image
    # Calculate the start and end indices for height and width
    start_y = (h - target_height) // 2
    start_x = (w - target_width) // 2
    end_y = start_y + target_height
    end_x = start_x + target_width

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]
    return cropped_image


def plot_road_boundary_with_fill(x_boundary_map):
    fig, ax = plt.subplots(figsize=(6, 6))
    canvas = FigureCanvas(fig)

    # Extract and sort x and y boundaries, swap roles for rotation
    y_values = sorted(x_boundary_map.keys())
    x_min_values = [x_boundary_map[y]['min'] for y in y_values]
    x_max_values = [x_boundary_map[y]['max'] for y in y_values]

    ax.fill_betweenx(y_values, x_min_values, x_max_values,
                     color='green', label='Road Area')
    ax.fill_betweenx(y_values, min(x_min_values) - 10, x_min_values,
                     color='red', label='Outside Road Area (Left)')
    ax.fill_betweenx(y_values, x_max_values, max(x_max_values) +
                     10, color='red', label='Outside Road Area (Right)')

    # X-coordinates (in swapped axes)
    teal_bbox_x = [-1.1, 1.1, 1.1, -1.1, -1.1]
    # Y-coordinates (in swapped axes)
    teal_bbox_y = [-2.4, -2.4, 2.4, 2.4, -2.4]
    ax.plot(teal_bbox_x, teal_bbox_y, color='teal',
            lw=2, label='Fixed Teal BBox')

    ax.axis('equal')  # Ensure equal scaling

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    ax.set_xlim(-30, 30)  # Adjust limits as needed for visualization
    ax.set_ylim(-30, 30)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')

    # Render the plot to a NumPy array
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
    image = image.reshape(int(height), int(width), 4)[:, :, :3]
    # Convert NumPy array to PIL Image and display it
    # image = Image.fromarray(image)
    # image.show()

    plt.close(fig)
    return image


def angle_between(v1, v2):
    return math.degrees(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))


def get_angle(car, wp):
    '''
    this function to find direction to selected waypoint
    '''
    vehicle_pos = car.get_transform()
    car_x = vehicle_pos.location.x
    car_y = vehicle_pos.location.y
    wp_x = wp.transform.location.x
    wp_y = wp.transform.location.y

    # vector to waypoint
    x = (wp_x - car_x)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5
    y = (wp_y - car_y)/((wp_y - car_y)**2 + (wp_x - car_x)**2)**0.5

    # car vector
    car_vector = vehicle_pos.get_forward_vector()
    degrees = angle_between((x, y), (car_vector.x, car_vector.y))

    return degrees
