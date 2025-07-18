import numpy as np
import matplotlib.pyplot as plt # For visualization
import open3d as o3d # For point cloud processing and ICP

# Load the data
try:
    extrinsics = np.load('extrinsics.npy')
    intrinsics = np.load('intrinsics.npy')
    depth_map = np.load('one-box.depth.npdata.npy')
    color_image = np.load('one-box.color.npdata.npy')
except FileNotFoundError:
    print("Ensure all .npy files are in the same directory as this script.")
    exit()

print("--- Data Loaded ---")
print("Extrinsics (T_WC or T_CW):\n", extrinsics)
print("\nIntrinsics (K):\n", intrinsics)
print("\nDepth Map Shape:", depth_map.shape)
print("Color Image Shape:", color_image.shape)

if len(color_image.shape) == 2:
    v_min, v_max = np.percentile(color_image[color_image > 0], [1, 99])
    color_image = np.clip(color_image, v_min, v_max)
    color_image = (color_image - v_min) / (v_max - v_min)
    color_image = np.stack([color_image] * 3, axis=-1)
    
# Visualize the depth and color images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Depth Map")
plt.imshow(depth_map, cmap='gray')
plt.colorbar(label='Depth (e.g., meters)')
plt.subplot(1, 2, 2)
plt.title("Color Image")
plt.imshow(color_image)
plt.show()


def depth_to_point_cloud(depth_map, intrinsics):
    # Get image dimensions
    height, width = depth_map.shape

    # Get intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    # Create arrays for pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Get depth values
    Z = depth_map

    # Filter out invalid depth values (e.g., 0 or very large/NaN)
    # Adjust this threshold based on your depth data's characteristics
    valid_mask = (Z > 0.01) & (Z < 10.0) # Example: > 1cm and < 10m

    # Apply the pinhole camera model equations
    X = (u[valid_mask] - cx) * Z[valid_mask] / fx
    Y = (v[valid_mask] - cy) * Z[valid_mask] / fy
    Z_valid = Z[valid_mask] # Use the filtered Z values

    # Combine into a point cloud (N, 3)
    points_camera_coords = np.vstack((X, Y, Z_valid)).T

    return points_camera_coords

# Generate the point cloud
points_3d_camera = depth_to_point_cloud(depth_map, intrinsics)
print("\n--- Point Cloud Generated ---")
print(f"Number of 3D points: {len(points_3d_camera)}")

# Visualize the point cloud using Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d_camera)
# Optionally, add colors from the color image if desired for visualization
# colors = color_image.reshape(-1, 3)[valid_mask] / 255.0 # Need to define valid_mask scope
# pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd], window_name="Raw Point Cloud")



# Assuming a rough depth range for the box. You might need to adjust these.
# Based on the previous visualization, estimate a reasonable range.
min_depth = 0.5  # Example: 50 cm
max_depth = 2.0  # Example: 2 meters

# Filter points by depth
filtered_points_depth = points_3d_camera[
    (points_3d_camera[:, 2] > min_depth) &
    (points_3d_camera[:, 2] < max_depth)
]

# Further filter using statistical outlier removal (Open3D)
# This helps remove isolated noisy points
pcd_filtered = o3d.geometry.PointCloud()
pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points_depth)

# Apply Statistical Outlier Removal
# nb_neighbors: Number of neighbors to consider for the average distance estimation.
# std_ratio: Standard deviation ratio. Points whose average distance is
# greater than the mean distance plus this factor times the standard
# deviation are removed.
cl, ind = pcd_filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
segmented_box_points = np.asarray(cl.points)

print("\n--- Point Cloud Filtered/Segmented ---")
print(f"Number of points after filtering: {len(segmented_box_points)}")

# Visualize the segmented point cloud
pcd_segmented = o3d.geometry.PointCloud()
pcd_segmented.points = o3d.utility.Vector3dVector(segmented_box_points)
pcd_segmented.paint_uniform_color([1, 0, 0]) # Paint it red for clear visualization
o3d.visualization.draw_geometries([pcd_segmented], window_name="Segmented Box Point Cloud")


# Define the dimensions of your known box (e.g., in meters)
box_length = 0.3  # X dimension
box_width = 0.2   # Y dimension
box_height = 0.15 # Z dimension

# Create a simple box model (8 corner points) centered at the origin
# This is our 'source' model that we want to align to the observed points
half_l, half_w, half_h = box_length / 2, box_width / 2, box_height / 2

# Corners of the box model
box_model_points = np.array([
    [-half_l, -half_w, -half_h],
    [ half_l, -half_w, -half_h],
    [-half_l,  half_w, -half_h],
    [ half_l,  half_w, -half_h],
    [-half_l, -half_w,  half_h],
    [ half_l, -half_w,  half_h],
    [-half_l,  half_w,  half_h],
    [ half_l,  half_w,  half_h]
])

print("\n--- Box Model Defined ---")
print("Box model points (8 corners):\n", box_model_points)

# Create an Open3D point cloud for the box model
pcd_box_model = o3d.geometry.PointCloud()
pcd_box_model.points = o3d.utility.Vector3dVector(box_model_points)
pcd_box_model.paint_uniform_color([0, 1, 0]) # Paint it green

# Optionally visualize the box model (will be very small if viewed alone)
# o3d.visualization.draw_geometries([pcd_box_model], window_name="Box Model (Green)")

# Define the dimensions of your known box (e.g., in meters)
box_length = 0.3  # X dimension
box_width = 0.2   # Y dimension
box_height = 0.15 # Z dimension

# Create a simple box model (8 corner points) centered at the origin
# This is our 'source' model that we want to align to the observed points
half_l, half_w, half_h = box_length / 2, box_width / 2, box_height / 2

# Corners of the box model
box_model_points = np.array([
    [-half_l, -half_w, -half_h],
    [ half_l, -half_w, -half_h],
    [-half_l,  half_w, -half_h],
    [ half_l,  half_w, -half_h],
    [-half_l, -half_w,  half_h],
    [ half_l, -half_w,  half_h],
    [-half_l,  half_w,  half_h],
    [ half_l,  half_w,  half_h]
])

print("\n--- Box Model Defined ---")
print("Box model points (8 corners):\n", box_model_points)

# Create an Open3D point cloud for the box model
pcd_box_model = o3d.geometry.PointCloud()
pcd_box_model.points = o3d.utility.Vector3dVector(box_model_points)
pcd_box_model.paint_uniform_color([0, 1, 0]) # Paint it green

# Optionally visualize the box model (will be very small if viewed alone)
# o3d.visualization.draw_geometries([pcd_box_model], window_name="Box Model (Green)")

# Create Open3D point cloud objects for ICP
source_pcd = pcd_box_model # Our box model
target_pcd = pcd_segmented # Our observed segmented points

# For better results, especially with ICP, it's good practice to estimate normals
# This helps in the correspondence finding and refinement
source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# ICP parameters
# threshold: Maximum correspondence distance
# initial_transformation: An initial guess for the transformation (identity if no prior knowledge)
# estimation_method: Which estimation method to use (Point-to-Point, Point-to-Plane)
# criteria: Convergence criteria
threshold = 0.1 # Max distance for point correspondences (e.g., 10 cm)
# A good initial guess is crucial for ICP. If the box is roughly upright,
# an identity matrix might be okay. Otherwise, you might need a coarse alignment first.
# Here, we start with an identity matrix as a simple initial guess.
initial_guess = np.identity(4)

# Run ICP
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd, threshold, initial_guess,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
)

# The result contains the estimated transformation
estimated_pose_transform = reg_p2p.transformation
print("\n--- Pose Estimation (ICP) Result ---")
print("Estimated Pose Transformation Matrix (T_camera_box):\n", estimated_pose_transform)

# Extract rotation matrix (R) and translation vector (t)
R_estimated = estimated_pose_transform[0:3, 0:3]
t_estimated = estimated_pose_transform[0:3, 3]

print("\nEstimated Rotation Matrix (R_camera_box):\n", R_estimated)
print("\nEstimated Translation Vector (t_camera_box) (X, Y, Z in camera coords):\n", t_estimated)

# Apply the estimated transformation to the source (box model) point cloud
transformed_box_pcd = source_pcd.transform(estimated_pose_transform)

# Visualize the original segmented points (target) and the transformed box model (source)
# This will show how well our estimated box pose aligns with the observed points
o3d.visualization.draw_geometries([target_pcd, transformed_box_pcd],
                                  window_name="Segmented Points (Red) and Transformed Box Model (Green)")

# --- More advanced visualization: Project transformed box onto the 2D image ---

def project_3d_to_2d(points_3d, intrinsics):
    # Apply intrinsic camera matrix K to 3D points
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

    # Project to 2D
    u = (points_3d[:, 0] * fx / points_3d[:, 2]) + cx
    v = (points_3d[:, 1] * fy / points_3d[:, 2]) + cy

    return np.vstack((u, v)).T

# Get the corner points of the transformed box
transformed_box_corners = np.asarray(transformed_box_pcd.points)

# Project these corners onto the 2D image plane
projected_corners_2d = project_3d_to_2d(transformed_box_corners, intrinsics)

# Visualize the color image with the projected box corners
plt.figure(figsize=(8, 6))
plt.imshow(color_image)
plt.title("Color Image with Projected Box Outline")

# Draw lines connecting the projected corners to form the box outline
# This part requires knowing the connectivity of the box corners.
# A standard box has 12 edges.
# Here's a simple way to draw the base and top, and vertical edges:
edges = [
    (0, 1), (0, 2), (1, 3), (2, 3),  # Bottom face
    (4, 5), (4, 6), (5, 7), (6, 7),  # Top face
    (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
]

for i, j in edges:
    if 0 <= projected_corners_2d[i, 0] < color_image.shape[1] and \
       0 <= projected_corners_2d[i, 1] < color_image.shape[0] and \
       0 <= projected_corners_2d[j, 0] < color_image.shape[1] and \
       0 <= projected_corners_2d[j, 1] < color_image.shape[0]:
        plt.plot([projected_corners_2d[i, 0], projected_corners_2d[j, 0]],
                 [projected_corners_2d[i, 1], projected_corners_2d[j, 1]],
                 'r-', linewidth=2) # Red line

plt.scatter(projected_corners_2d[:, 0], projected_corners_2d[:, 1], color='blue', marker='o', s=50) # Mark corners
plt.xlim(0, color_image.shape[1])
plt.ylim(color_image.shape[0], 0) # Invert y-axis to match image coordinates
plt.show()

print("\n--- Pose Estimation Complete and Visualized ---")