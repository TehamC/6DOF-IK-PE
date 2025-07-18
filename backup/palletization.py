import pybullet as p
import pybullet_data
import time
import os
import numpy as np
import imageio

# Connect to GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)

# Load plane and robot
p.loadURDF("plane.urdf")
combined_urdf_path = os.path.join("robot-urdfs", "combined_robot.urdf")
robot_id = p.loadURDF(
    combined_urdf_path,
    basePosition=[0, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=p.URDF_USE_SELF_COLLISION
)

# Get joint info
num_joints = p.getNumJoints(robot_id)
movable_joints = []
joint_name_to_index = {}
linear_axis_joint = None
abb_joints = []

for i in range(num_joints):
    joint_info = p.getJointInfo(robot_id, i)
    joint_type = joint_info[2]
    joint_name = joint_info[1].decode()
    if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
        movable_joints.append(i)
        joint_name_to_index[joint_name] = i
        if "linear_axis" in joint_name.lower():
            linear_axis_joint = i
        else:
            abb_joints.append(i)

# Verify EE link
ee_link_index = 8  # abb_flange
ee_link_info = p.getLinkState(robot_id, ee_link_index)
ee_pos = [round(x, 2) for x in ee_link_info[0]]
print(f"EE Link Index: {ee_link_index}, Name: {p.getJointInfo(robot_id, ee_link_index)[12].decode()}, Initial Position: {ee_pos}")

# Define poses with rounded coordinates
goal_poses = [
    ([5.0, 1.63, 1.5], [0.0, 3.14, 0.0]),
    ([2.74, -1.16, 0.92], [0.0, 3.14, 0.0]),
    ([4.84, -1.16, 2.11], [0.0, 3.14, 0.0]),
    ([3.63, 1.0, 1.71], [0.0, 3.14, 0.0]),
    ([2.53, -2.16, 1.71], [0.0, 3.14, 0.0])
]
soft_reset_rpy = [0.0, 3.14, 0.0]
soft_reset_yz = [0.0, 1.5]

# Define initial pose
initial_position = [0.0, 0.0, 1.5]
initial_orientation = p.getQuaternionFromEuler([0.0, 3.14, 0.0])

# Initialize joint positions
initial_ik_solution = p.calculateInverseKinematics(
    bodyUniqueId=robot_id,
    endEffectorLinkIndex=ee_link_index,
    targetPosition=initial_position,
    targetOrientation=initial_orientation,
    maxNumIterations=200,
    residualThreshold=1e-5
)
if len(initial_ik_solution) < len(movable_joints):
    print(f"Warning: Initial IK solution length ({len(initial_ik_solution)}) is less than movable joints ({len(movable_joints)}).")
    p.disconnect()
    exit()
initial_joint_positions = [round(x, 2) for x in initial_ik_solution[:len(movable_joints)]]
for i, joint_index in enumerate(movable_joints):
    p.resetJointState(robot_id, joint_index, initial_joint_positions[i])

# Visualize initial pose
target_axis_length = 0.4
target_axis_width = 3
initial_rot_matrix = p.getMatrixFromQuaternion(initial_orientation)
initial_x_axis_vec = [initial_rot_matrix[0], initial_rot_matrix[3], initial_rot_matrix[6]]
initial_y_axis_vec = [initial_rot_matrix[1], initial_rot_matrix[4], initial_rot_matrix[7]]
initial_z_axis_vec = [initial_rot_matrix[2], initial_rot_matrix[5], initial_rot_matrix[8]]
p.addUserDebugLine(initial_position,
                   [initial_position[i] + target_axis_length * initial_x_axis_vec[i] for i in range(3)],
                   [0, 1, 1], lineWidth=target_axis_width, lifeTime=0)
p.addUserDebugLine(initial_position,
                   [initial_position[i] + target_axis_length * initial_y_axis_vec[i] for i in range(3)],
                   [0, 1, 1], lineWidth=target_axis_width, lifeTime=0)
p.addUserDebugLine(initial_position,
                   [initial_position[i] + target_axis_length * initial_z_axis_vec[i] for i in range(3)],
                   [0, 1, 1], lineWidth=target_axis_width, lifeTime=0)

# Initialize goal pose visualization
current_goal_position = goal_poses[0][0].copy()
current_goal_rpy = goal_poses[0][1].copy()
goal_orientation = p.getQuaternionFromEuler(current_goal_rpy)
goal_rot_matrix = p.getMatrixFromQuaternion(goal_orientation)
goal_x_axis_vec = [goal_rot_matrix[0], goal_rot_matrix[3], goal_rot_matrix[6]]
goal_y_axis_vec = [goal_rot_matrix[1], goal_rot_matrix[4], goal_rot_matrix[7]]
goal_z_axis_vec = [goal_rot_matrix[2], goal_rot_matrix[5], goal_rot_matrix[8]]
goal_x_line = p.addUserDebugLine(current_goal_position,
                                 [current_goal_position[i] + target_axis_length * goal_x_axis_vec[i] for i in range(3)],
                                 [1, 0, 0], lineWidth=target_axis_width, lifeTime=0)
goal_y_line = p.addUserDebugLine(current_goal_position,
                                 [current_goal_position[i] + target_axis_length * goal_y_axis_vec[i] for i in range(3)],
                                 [0, 1, 0], lineWidth=target_axis_width, lifeTime=0)
goal_z_line = p.addUserDebugLine(current_goal_position,
                                 [current_goal_position[i] + target_axis_length * goal_z_axis_vec[i] for i in range(3)],
                                 [0, 0, 1], lineWidth=target_axis_width, lifeTime=0)

# Initialize EE axes
ee_x_line = None
ee_y_line = None
ee_z_line = None

print(f"Initial Pose (X,Y,Z): {[round(x, 2) for x in initial_position]}, RPY: {[0.0, 3.14, 0.0]}")
print(f"Initial Goal Pose (X,Y,Z): {[round(x, 2) for x in current_goal_position]}, RPY: {[0.0, 3.14, 0.0]}")

# Setup frame saving
output_dir = "animation_frames"
os.makedirs(output_dir, exist_ok=True)
frame_index = 0

def save_frame():
    global frame_index
    try:
        img = p.getCameraImage(640, 360, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        img = np.reshape(img, (360, 640, 4))[:, :, :3]
        imageio.imwrite(os.path.join(output_dir, f"frame_{frame_index:04d}.png"), img)
        frame_index += 1
    except Exception as e:
        print(f"Error saving frame: {str(e)}")

# Adjust camera
def update_camera(ee_pos, goal_pos):
    try:
        midpoint = [(ee_pos[i] + goal_pos[i]) / 2 for i in range(3)]
        distance = max(4.0, np.linalg.norm(np.array(ee_pos) - np.array(goal_pos)) * 1.5)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=50,
            cameraPitch=-30,
            cameraTargetPosition=midpoint
        )
    except Exception as e:
        print(f"Error updating camera: {str(e)}")

# Move to pose function
def move_to_pose(target_position, target_rpy, pose_name):
    global initial_joint_positions, ee_x_line, ee_y_line, ee_z_line
    target_position = [round(x, 2) for x in target_position]
    target_rpy = [round(x, 2) for x in target_rpy]
    target_orientation = p.getQuaternionFromEuler(target_rpy)
    
    # Update goal visualization
    global goal_x_line, goal_y_line, goal_z_line
    try:
        p.removeUserDebugItem(goal_x_line)
        p.removeUserDebugItem(goal_y_line)
        p.removeUserDebugItem(goal_z_line)
    except Exception as e:
        print(f"Error removing goal debug lines: {str(e)}")
    goal_rot_matrix = p.getMatrixFromQuaternion(target_orientation)
    goal_x_axis_vec = [goal_rot_matrix[0], goal_rot_matrix[3], goal_rot_matrix[6]]
    goal_y_axis_vec = [goal_rot_matrix[1], goal_rot_matrix[4], initial_rot_matrix[7]]
    goal_z_axis_vec = [goal_rot_matrix[2], goal_rot_matrix[5], goal_rot_matrix[8]]
    goal_x_line = p.addUserDebugLine(target_position,
                                     [target_position[i] + target_axis_length * goal_x_axis_vec[i] for i in range(3)],
                                     [1, 0, 0], lineWidth=target_axis_width, lifeTime=0)
    goal_y_line = p.addUserDebugLine(target_position,
                                     [target_position[i] + target_axis_length * goal_y_axis_vec[i] for i in range(3)],
                                     [0, 1, 0], lineWidth=target_axis_width, lifeTime=0)
    goal_z_line = p.addUserDebugLine(target_position,
                                     [target_position[i] + target_axis_length * goal_z_axis_vec[i] for i in range(3)],
                                     [0, 0, 1], lineWidth=target_axis_width, lifeTime=0)
    
    # Update camera
    ee_state = p.getLinkState(robot_id, ee_link_index)
    update_camera(ee_state[0], target_position)
    
    print(f"--- Moving to {pose_name} (X,Y,Z): {target_position}, RPY: {target_rpy} ---")
    
    # Compute IK solution
    try:
        ik_solution = p.calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=ee_link_index,
            targetPosition=target_position,
            targetOrientation=target_orientation,
            maxNumIterations=200,
            residualThreshold=1e-5
        )
        if len(ik_solution) < len(movable_joints):
            print(f"Warning: IK solution length ({len(ik_solution)}) is less than movable joints ({len(movable_joints)}).")
            return False
    except Exception as e:
        print(f"IK failed for {pose_name}: {str(e)}")
        return False
    
    goal_joint_positions = [round(x, 2) for x in ik_solution[:len(movable_joints)]]
    print(f"IK Solution for {pose_name}: {goal_joint_positions}")
    
    # Move all joints
    for i, joint_index in enumerate(movable_joints):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=goal_joint_positions[i],
            force=2000,
            positionGain=0.05,
            velocityGain=1.5,
            maxVelocity=3.0
        )
    for _ in range(480):
        p.stepSimulation()
        save_frame()
        time.sleep(1.0 / 240.0)
    
    # Lock final pose
    print(f"--- Locking {pose_name} ---")
    for i, joint_index in enumerate(movable_joints):
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=goal_joint_positions[i],
            force=3000,
            positionGain=0.05,
            velocityGain=1.5,
            maxVelocity=3.0
        )
    for _ in range(240):
        p.stepSimulation()
        save_frame()
        time.sleep(1.0 / 240.0)
    
    # Verify joint states
    current_joint_states = [round(p.getJointState(robot_id, i)[0], 2) for i in movable_joints]
    print(f"Current Joint States: {current_joint_states}")
    
    # Visualize EE axes
    if ee_x_line is not None:
        p.removeUserDebugItem(ee_x_line)
        p.removeUserDebugItem(ee_y_line)
        p.removeUserDebugItem(ee_z_line)
    ee_state = p.getLinkState(robot_id, ee_link_index)
    ee_pos = [round(x, 2) for x in ee_state[0]]
    ee_orient = ee_state[1]
    ee_rot_matrix = p.getMatrixFromQuaternion(ee_orient)
    ee_x_axis_vec = [ee_rot_matrix[0], ee_rot_matrix[3], ee_rot_matrix[6]]
    ee_y_axis_vec = [ee_rot_matrix[1], ee_rot_matrix[4], ee_rot_matrix[7]]
    ee_z_axis_vec = [ee_rot_matrix[2], ee_rot_matrix[5], ee_rot_matrix[8]]
    ee_x_line = p.addUserDebugLine(ee_pos,
                                   [ee_pos[i] + target_axis_length * ee_x_axis_vec[i] for i in range(3)],
                                   [1, 0, 1], lineWidth=target_axis_width, lifeTime=0)
    ee_y_line = p.addUserDebugLine(ee_pos,
                                   [ee_pos[i] + target_axis_length * ee_y_axis_vec[i] for i in range(3)],
                                   [1, 1, 0], lineWidth=target_axis_width, lifeTime=0)
    ee_z_line = p.addUserDebugLine(ee_pos,
                                   [ee_pos[i] + target_axis_length * ee_z_axis_vec[i] for i in range(3)],
                                   [0, 1, 1], lineWidth=target_axis_width, lifeTime=0)
    ee_rpy = [round(x, 2) for x in p.getEulerFromQuaternion(ee_orient)]
    print(f"EE Pose (X,Y,Z): {ee_pos}, RPY: {ee_rpy}")
    print("--- EE Axes Drawn (Magenta=X, Yellow=Y, Cyan=Z) ---")
    
    # Update initial_joint_positions
    initial_joint_positions = goal_joint_positions
    return True

# Run animation sequence
p.setRealTimeSimulation(0)

# Move to initial pose
print("--- Moving to Initial Pose ---")
if not move_to_pose(initial_position, [0.0, 3.14, 0.0], "Initial Pose"):
    print("Failed to set initial pose. Exiting.")
    p.disconnect()
    exit()

# Iterate through goal poses and soft resets
for i, (goal_pos, goal_rpy) in enumerate(goal_poses):
    # Move to goal pose
    if not move_to_pose(goal_pos, goal_rpy, f"Goal Pose {i+1}"):
        print(f"Failed to reach Goal Pose {i+1}. Skipping.")
        continue
    
    # Move to soft reset pose
    soft_reset_pos = [round(goal_pos[0], 2), round(soft_reset_yz[0], 2), round(soft_reset_yz[1], 2)]
    if not move_to_pose(soft_reset_pos, soft_reset_rpy, f"Soft Reset {i+1}"):
        print(f"Failed to reach Soft Reset {i+1}. Skipping.")
        continue

# End simulation
p.disconnect()
print("Simulation ended.")