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
depth_map = np.load("one-box.depth.npdata.npy")
color_image = np.load("one-box.color.npdata.npy")
extrinsics = np.load("extrinsics.npy")  # Load extrinsics


# Normalize and convert to RGB with better contrast
if len(color_image.shape) == 2:
    v_min, v_max = np.percentile(color_image[color_image > 0], [1, 99])
    color_image = np.clip(color_image, v_min, v_max)
    color_image = (color_image - v_min) / (v_max - v_min)
    color_image = np.stack([color_image] * 3, axis=-1)




def select_nearest_cluster(depth_map, intrinsics, eps=0.05, min_points=20, depth_scale=1.0):
    """
    Select the cluster with the lowest mean z-value (closest to camera).
    
    Args:
        depth_map (np.ndarray): Depth map (H, W).
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        eps (float): DBSCAN clustering distance (default 0.05).
        min_points (int): Minimum points per cluster (default 20).
        depth_scale (float): Depth map scale (default 1.0 for meters).
    
    Returns:
        np.ndarray: (N, 3) array of 3D points for the selected cluster.
        float: Mean z-value of the selected cluster.
        int: Selected cluster label.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth_map.shape

    print(f"Depth map shape: {depth_map.shape}, min: {depth_map.min():.4f}, max: {depth_map.max():.4f}")

    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intrinsic, depth_scale=depth_scale, depth_trunc=4.0
    )
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() < 0:
        raise ValueError("No clusters found")

    points = np.asarray(pcd.points)
    mean_z_per_cluster = {
        label: np.mean(points[labels == label][:, 2])
        for label in np.unique(labels) if label != -1
    }

    selected_label = min(mean_z_per_cluster, key=mean_z_per_cluster.get)
    selected_points = points[labels == selected_label]
    mean_z = mean_z_per_cluster[selected_label]

    print(f"Selected cluster: Label {selected_label}, Mean Z {mean_z:.4f} m, Points {len(selected_points)}")
    return selected_points, mean_z, selected_label

def filter_cluster_to_depth_coords(cluster_points, mean_z, intrinsics, z_threshold=0.1):
    """
    Filters 3D cluster points near mean_z and converts to [u, v, z] image coordinates.
    
    Args:
        cluster_points (np.ndarray): (N, 3) array of 3D points.
        mean_z (float): Mean z-value of the cluster.
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        z_threshold (float): Z-value threshold for filtering (default 0.1).
    
    Returns:
        np.ndarray: (N, 3) array of [x_pixel, y_pixel, z].
    """
    close_mask = np.abs(cluster_points[:, 2] - mean_z) < z_threshold
    filtered_points = cluster_points[close_mask]

    if len(filtered_points) == 0:
        return np.empty((0, 3))

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x, y, z = filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2]
    u = (fx * x / z + cx).astype(float)
    v = (fy * y / z + cy).astype(float)
    return np.stack([u, v, z], axis=1)

def sort_corners_clockwise_2d(corners_2d):
    """
    Sort 2D corners clockwise around their centroid.

    Args:
        corners_2d (np.ndarray): (4, 2) array of [pixel_x, pixel_y].

    Returns:
        np.ndarray: Indices sorting corners clockwise.
    """
    centroid = np.mean(corners_2d, axis=0)
    angles = np.arctan2(corners_2d[:, 1] - centroid[1], corners_2d[:, 0] - centroid[0])
    return np.argsort(angles)

def get_rectangle_2d_depth_points_pcl(filtered_data, color_image=None, plot_2d=True):
    """
    Calculate 2D minimum area bounding rectangle and return corners with depth.

    Args:
        filtered_data (np.ndarray): (N, 3) array of [x_pixel, y_pixel, z].
        color_image (np.ndarray, optional): Color image (H, W, 3) for visualization.
        plot_2d (bool): If True, plots the 2D rectangle and points.

    Returns:
        np.ndarray: (4, 3) array of [x_pixel, y_pixel, z] for rectangle corners.
    """
    filtered_x, filtered_y, filtered_z = filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2]
    points_xy = np.column_stack((filtered_x, filtered_y)).astype(np.float32)

    if len(points_xy) < 2:
        print("Not enough points to form a rectangle.")
        return np.empty((0, 3))

    rect = cv2.minAreaRect(points_xy)
    (center_x_2d, center_y_2d), (width, height), angle = rect
    area = width * height
    print(f"Rectangle Center (2D): ({center_x_2d:.2f}, {center_y_2d:.2f})")
    print(f"Rectangle Dimensions (Width, Height): ({width:.2f}, {height:.2f})")
    print(f"Rectangle Angle (degrees): {angle:.2f}")
    print(f"Rectangle Area: {area:.2f}")

    box_2d = cv2.boxPoints(rect)
    sorted_indices = sort_corners_clockwise_2d(box_2d)
    box_2d = box_2d[sorted_indices]

    kdtree = KDTree(points_xy)
    corners_2d_depth = []
    for corner_2d in box_2d:
        _, idx = kdtree.query(corner_2d.reshape(1, -1))
        corner_data = filtered_data[idx[0]]  # [u, v, z]
        corners_2d_depth.append(corner_data)
    corners_2d_depth = np.array(corners_2d_depth)
    # print(f"2D corners with depth:\n{corners_2d_depth}")

    if plot_2d:
        plt.figure(figsize=(10, 8))
        if color_image is not None:
            plt.imshow(color_image)
        plt.scatter(filtered_x, filtered_y, s=10, alpha=0.7, color='blue', label='Filtered Points')
        plt.plot(np.append(box_2d[:, 0], box_2d[0, 0]), np.append(box_2d[:, 1], box_2d[0, 1]), 
                color='red', linestyle='-', linewidth=2, label='Bounding Box')
        plt.scatter(box_2d[:, 0], box_2d[:, 1], s=100, color='purple', marker='o', label='Corners')
        plt.scatter(center_x_2d, center_y_2d, s=100, color='green', marker='X', label='Center')
        for i, (x, y) in enumerate(box_2d):
            plt.text(x + 5, y + 5, f'({x:.0f}, {y:.0f})', color='purple', fontsize=9)
        plt.text(center_x_2d + 5, center_y_2d + 5, f'({center_x_2d:.0f}, {center_y_2d:.0f})', 
                color='green', fontsize=9)
        plt.gca().invert_yaxis()
        plt.title('PCL Filtered Points with Bounding Box')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    return corners_2d_depth

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

    # Color points based on z-value
    points = np.asarray(pcd.points)
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
    colormap = plt.get_cmap("viridis")
    colors = colormap(z_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
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


# Normalize and convert to RGB with better contrast
if len(color_image.shape) == 2:
    v_min, v_max = np.percentile(color_image[color_image > 0], [1, 99])
    color_image = np.clip(color_image, v_min, v_max)
    color_image = (color_image - v_min) / (v_max - v_min)
    color_image = np.stack([color_image] * 3, axis=-1)
    
# PCL approach
cluster_points, mean_z, selected_label = select_nearest_cluster(depth_map, intrinsics, depth_scale=1.0)
filtered_data = filter_cluster_to_depth_coords(cluster_points, mean_z, intrinsics, z_threshold=0.1)
top_corners_2d_depth = get_rectangle_2d_depth_points_pcl(filtered_data, color_image=color_image, plot_2d=True)



T_camera_object = estimate_box_pose(top_corners_2d_depth, intrinsics, box_size=0.3)
print("Estimated T_camera_object:\n", T_camera_object)
plot_pointcloud_with_3d_box(depth_map, intrinsics, extrinsics, T_camera_object, box_size=0.3)


def plot_pointcloud_only(depth_map, intrinsics, extrinsics):
    """
    Plots the point cloud generated from a depth map, colored by Z-coordinate.

    Args:
        depth_map (np.ndarray): Depth map (H, W).
        intrinsics (np.ndarray): 3x3 camera intrinsic matrix.
        extrinsics (np.ndarray): 4x4 camera extrinsic matrix (camera to world).
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth_map.shape

    depth_o3d = o3d.geometry.Image(depth_map.astype(np.float32))
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=fx, fy=fy, cx=cx, cy=cy)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, o3d_intrinsic, depth_scale=1.0, depth_trunc=4.0
    )
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Color points based on z-value
    points = np.asarray(pcd.points)
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    z_normalized = (z_values - z_min) / (z_max - z_min + 1e-6)
    colormap = plt.get_cmap("viridis")
    colors = colormap(z_normalized)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Transform the point cloud to the world frame using extrinsics
    pcd.transform(extrinsics)

    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud without Box (Z-colored)")

plot_pointcloud_only(depth_map, intrinsics, extrinsics)