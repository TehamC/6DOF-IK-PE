import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import DBSCAN
import cv2 # OpenCV library
import os
from scipy.spatial import KDTree
os.environ['XDG_SESSION_TYPE'] = 'x11'

# Load provided files
intrinsics = np.load("intrinsics.npy")
extrinsics = np.load("extrinsics.npy")
depth_map = np.load("one-box.depth.npdata.npy")
color_image = np.load("one-box.color.npdata.npy")

# Print shape for debugging
print("Color image shape:", color_image.shape)

# Normalize and convert to RGB with better contrast
if len(color_image.shape) == 2:
    v_min, v_max = np.percentile(color_image[color_image > 0], [1, 99])
    color_image = np.clip(color_image, v_min, v_max)
    color_image = (color_image - v_min) / (v_max - v_min)
    color_image = np.stack([color_image] * 3, axis=-1)




def get_box_points(depth_map, intrinsics, eps=0.05, min_points=20):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth_map.shape

    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intrinsic, depth_scale=1000.0, depth_trunc=4000.0
    )
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        raise ValueError("No clusters found")

    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise label

    min_mean_depth = np.inf
    closest_label = None

    points_np = np.asarray(pcd.points)

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_points = points_np[cluster_indices]

        mean_depth = np.mean(cluster_points[:, 2])  # Z coordinate is depth
        if mean_depth < min_mean_depth:
            min_mean_depth = mean_depth
            closest_label = label

    # Extract points for the closest cluster
    closest_indices = np.where(labels == closest_label)[0]
    closest_points = points_np[closest_indices]
    return closest_points, closest_label

def visualize_points(points, color=[1, 0, 0]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    o3d.visualization.draw_geometries([pcd], window_name="Closest Cluster Points")

# Usage
intrinsics = np.load("intrinsics.npy")
depth_map = np.load("one-box.depth.npdata.npy")

box_points, cluster_label = get_box_points(depth_map, intrinsics)
print(f"Closest cluster label: {cluster_label}, points count: {len(box_points)}")

visualize_points(box_points)


def points3d_to_pixel_depth(points_3d, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]

    # Project to pixel coordinates (u, v)
    u = (X * fx) / Z + cx
    v = (Y * fy) / Z + cy

    # Stack as (N,3): pixel_x, pixel_y, depth
    result = np.stack([u, v, Z], axis=1)

    return result

# Example usage:
# closest_points is your selected 3D points in camera coordinates, shape (N,3)
box_uvz = points3d_to_pixel_depth(box_points, intrinsics)


def get_rectangle_3d_points(filtered_data, color_image=None, plot_2d=True):
    """
    Calculates the 2D minimum area bounding rectangle from filtered points,
    and retrieves corresponding depth values from the filtered_data itself
    using nearest neighbor search.

    Args:
        filtered_data (np.ndarray): A NumPy array of shape (N, 3) where each row
                                    is [y, x, depth] for the points belonging to
                                    the detected object (e.g., from DBSCAN).
        color_image (np.ndarray, optional): The original color image (H, W, 3)
                                            for 2D visualization background. Defaults to None.
        plot_2d (bool, optional): If True, a 2D plot showing the filtered points
                                  and the rectangle with corners/midpoints will be displayed.
                                  Defaults to True.

    Returns:
        tuple: A tuple containing:
            - corners_3d (np.ndarray): (4, 3) array of [x, y, depth] for the rectangle's corners.
            - midpoints_3d (np.ndarray): (4, 3) array of [x, y, depth] for the midpoints of each side.
            - rect_center_3d (np.ndarray): (1, 3) array of [x, y, depth] for the rectangle's center.
            - rect_dims (tuple): (width, height, angle) of the 2D bounding rectangle.
        Returns (None, None, None, None) if not enough points are provided to form a rectangle.
    """
    # Separate y, x, and depth from filtered_data
    filtered_y = filtered_data[:, 0]
    filtered_x = filtered_data[:, 1]
    filtered_depth = filtered_data[:, 2] # We now use this directly!
    
    # Create a KDTree for efficient nearest neighbor lookup on the filtered (y, x) coordinates
    filtered_spatial_points = np.column_stack((filtered_y, filtered_x))
    kdtree = KDTree(filtered_spatial_points)

    # points_xy is (x, y) for OpenCV minAreaRect
    points_xy_for_min_area_rect = np.column_stack((filtered_x, filtered_y)).astype(np.float32)

    if len(points_xy_for_min_area_rect) < 2:
        print("Not enough points to form a rectangle. Need at least 2 points.")
        return None, None, None, None

    # Calculate the minimum area rotated rectangle
    rect = cv2.minAreaRect(points_xy_for_min_area_rect)
    (center_x_2d, center_y_2d), (width, height), angle = rect

    print(f"Rectangle Center (2D): ({center_x_2d:.2f}, {center_y_2d:.2f})")
    print(f"Rectangle Dimensions (Width, Height): ({width:.2f}, {height:.2f})")
    print(f"Rectangle Angle (degrees): {angle:.2f}")

    # Get the four corners of the rotated rectangle (x, y) pixel coordinates
    box_2d = cv2.boxPoints(rect)
    corners_2d_float = box_2d # Keep float for precise lookup
    
    # --- Retrieve Depth Information for Rectangle Points using filtered_data ---

    # Helper to get depth using KDTree for a given (x, y) query point
    def get_depth_from_filtered_data(query_x, query_y):
        query_point_yx = np.array([query_y, query_x]) # KDTree was built on (y, x)
        distance, index = kdtree.query(query_point_yx)
        
        # Check if the closest point is reasonably close (optional thresholding)
        # If distance > some_threshold, it might mean the queried point is far from any filtered point.
        # For now, we'll just return the depth of the closest point.
        return filtered_depth[index]

    # 1. Get 3D Corners
    corners_3d = []
    print("\nCorner Points (X, Y, Depth):")
    for i, corner_2d in enumerate(corners_2d_float):
        x, y = corner_2d[1], corner_2d[0]
        depth_val = get_depth_from_filtered_data(x, y)
        corners_3d.append([x, y, depth_val])
        print(f"Corner {i+1}: ({x:.2f}, {y:.2f}, Depth: {depth_val:.3f})")
    corners_3d = np.array(corners_3d)

    # 2. Get 3D Midpoints
    midpoints_2d_float = [] # Store float midpoints for 2D plotting
    midpoints_3d = []
    print("\nMidpoints of each side (X, Y, Depth):")
    for i in range(4):
        p1 = corners_2d_float[i]
        p2 = corners_2d_float[(i + 1) % 4]
        
        mid_x_float = (p1[0] + p2[0]) / 2
        mid_y_float = (p1[1] + p2[1]) / 2
        midpoints_2d_float.append([mid_x_float, mid_y_float])
        
        depth_val = get_depth_from_filtered_data(mid_x_float, mid_y_float)
        midpoints_3d.append([mid_x_float, mid_y_float, depth_val])
        print(f"Midpoint {i+1}: ({mid_x_float:.2f}, {mid_y_float:.2f}, Depth: {depth_val:.3f})")
    midpoints_2d_float = np.array(midpoints_2d_float)
    midpoints_3d = np.array(midpoints_3d)

    # 3. Get 3D Center
    center_depth_val = get_depth_from_filtered_data(center_x_2d, center_y_2d)
    rect_center_3d = np.array([[center_x_2d, center_y_2d, center_depth_val]])
    print(f"\nRectangle Center (X, Y, Depth): ({rect_center_3d[0,0]:.2f}, {rect_center_3d[0,1]:.2f}, Depth: {rect_center_3d[0,2]:.3f})")


    # --- 2D Visualization ---
    if plot_2d:
        plt.figure(figsize=(10, 8))
        if color_image is not None:
            plt.imshow(color_image)
        
        plt.scatter(filtered_x, filtered_y, s=10, alpha=0.7, color='blue', label='Filtered Points')
        
        # Plot the bounding box using float 2D corners for plotting consistency
        plt.plot(corners_2d_float[:, 0], corners_2d_float[:, 1], color='red', linestyle='-', linewidth=2, label='Oriented Bounding Box')
        plt.plot([corners_2d_float[-1, 0], corners_2d_float[0, 0]], [corners_2d_float[-1, 1], corners_2d_float[0, 1]], color='red', linestyle='-', linewidth=2)
        
        plt.scatter(corners_2d_float[:, 0], corners_2d_float[:, 1], s=100, color='purple', marker='o', label='Corner Points')
        plt.scatter(midpoints_2d_float[:, 0], midpoints_2d_float[:, 1], s=120, color='orange', marker='s', label='Midpoints')
        plt.scatter(center_x_2d, center_y_2d, s=100, color='green', marker='X', label='Rectangle Center')

        ax = plt.gca()
        ax.invert_yaxis()
        
        all_x_plot = np.concatenate((filtered_x, corners_2d_float[:,0], midpoints_2d_float[:,0], [center_x_2d]))
        all_y_plot = np.concatenate((filtered_y, corners_2d_float[:,1], midpoints_2d_float[:,1], [center_y_2d]))

        ax.set_xlim(min(0, np.min(all_x_plot)-10), np.max(all_x_plot)+10)
        ax.set_ylim(np.max(all_y_plot)+10, min(0, np.min(all_y_plot)-10))

        plt.title('Filtered Points with Bounding Box, Corners, and Midpoints')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate (Increases Downwards)')
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    return corners_3d, midpoints_3d, rect_center_3d, (width, height, angle)


rectangle_cp = get_rectangle_3d_points(box_uvz)
print("cp: ",rectangle_cp[0])


def sort_corners_clockwise_2d(corners_2d):
    """
    Sort 4 corners in clockwise order starting from top-left in 2D image space.

    Args:
        corners_2d (np.ndarray): shape (4, 2)

    Returns:
        np.ndarray: indices for sorted corners
    """
    center = np.mean(corners_2d, axis=0)
    angles = np.arctan2(corners_2d[:, 1] - center[1], corners_2d[:, 0] - center[0])
    return np.argsort(angles)

def estimate_box_pose(top_corners_2d_depth, intrinsics, box_size=0.3):
    """
    Estimate 4x4 transformation matrix (camera to object frame) for a box.

    Args:
        top_corners_2d_depth (np.ndarray): (4, 3) array of [pixel_x, pixel_y, depth].
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        box_size (float): Size of the cube (default 0.3m).

    Returns:
        np.ndarray: 4x4 transformation matrix (camera to object frame).
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # --- Sort corners by 2D pixel positions to get consistent orientation ---
    sorted_indices = sort_corners_clockwise_2d(top_corners_2d_depth[:, :2])
    sorted_corners = top_corners_2d_depth[sorted_indices]

    # --- Project to 3D camera coordinates ---
    top_corners_camera = np.zeros((4, 3))
    for i in range(4):
        u, v, depth = sorted_corners[i]
        if depth <= 0:
            raise ValueError("Invalid depth value for corner {}".format(i))
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        top_corners_camera[i] = [x, y, z]

    # --- Define box corners in object frame (top face, Z = -half_size) ---
    half_size = box_size / 2
    corners_object = np.array([
        [-half_size, -half_size, -half_size],  # Corner 0
        [half_size, -half_size, -half_size],   # Corner 1
        [half_size, half_size, -half_size],    # Corner 2
        [-half_size, half_size, -half_size]    # Corner 3
    ])

    # --- Compute transformation ---
    center_object = np.mean(corners_object, axis=0)
    center_camera = np.mean(top_corners_camera, axis=0)

    centered_object = corners_object - center_object
    centered_camera = top_corners_camera - center_camera

    H = centered_object.T @ centered_camera
    U, _, Vt = np.linalg.svd(H)
    rotation = Vt.T @ U.T
    if np.linalg.det(rotation) < 0:
        Vt[2, :] *= -1
        rotation = Vt.T @ U.T

    translation = center_camera - rotation @ center_object

    T_camera_object = np.eye(4)
    T_camera_object[:3, :3] = rotation
    T_camera_object[:3, 3] = translation

    return T_camera_object

def plot_pointcloud_with_3d_box(depth_map, intrinsics, extrinsics, box_pose, box_size=0.3):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth_map.shape

    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intrinsic, depth_scale=1.0, depth_trunc=4.0
    )
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    pcd.transform(extrinsics)

    half_size = box_size / 2
    vertices_obj = np.array([
        [-half_size, -half_size, -half_size], [half_size, -half_size, -half_size],
        [half_size, half_size, -half_size], [-half_size, half_size, -half_size],
        [-half_size, -half_size, half_size], [half_size, -half_size, half_size],
        [half_size, half_size, half_size], [-half_size, half_size, half_size]
    ])
    vertices_hom = np.hstack((vertices_obj, np.ones((8, 1))))
    vertices_camera = (vertices_hom @ box_pose.T)[:, :3]
    vertices_world = vertices_camera @ extrinsics[:3, :3].T + extrinsics[:3, 3]

    triangles = o3d.utility.Vector3iVector([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 1], [1, 4, 5], [1, 5, 2], [2, 5, 6],
        [2, 6, 3], [3, 6, 7], [3, 7, 0], [0, 7, 4]
    ])
    box_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices_world), triangles)
    box_mesh.paint_uniform_color([0.0, 1.0, 0.0])
    box_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([pcd, box_mesh], window_name="Point Cloud with 3D Box")

# --- Usage ---

top_corners_2d_depth = np.array([
    [422.32, 677.00, 2.33],
    [603.81, 433.54, 2.347],
    [842.14, 611.22, 2.34],
    [660.65, 854.67, 2.32]
])

top_corners_2d_depth = np.array([
    [421.45535278, 679.29864502, 2.35619187],
    [603.08703613, 432.16033936, 2.33706594],
    [843.49768066, 608.84759521, 2.31714606],
    [661.8659668, 855.98590088, 2.34085774]
])

top_corners_2d_depth = np.array([
    [602.99974389, 434.62495485,   2.33650306],
    [834.41708736, 609.79188424,   2.31349824],
    [661.50005501, 853.0,           2.34036613],
    [438.69687305, 678.43953324,   2.35630706]
])

T_camera_object = estimate_box_pose(top_corners_2d_depth, intrinsics, box_size=0.3)
# T_camera_object = estimate_box_pose(rectangle_cp[0], intrinsics, box_size=0.3)
print("Estimated T_camera_object:\n", T_camera_object)

plot_pointcloud_with_3d_box(depth_map, intrinsics, extrinsics, T_camera_object, box_size=0.3)