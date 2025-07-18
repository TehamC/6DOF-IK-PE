import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from IK_solver_plx import calculateInverseKinematics

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
ee_pos = ee_link_info[0]
print(f"EE Link Index: {ee_link_index}, Name: {p.getJointInfo(robot_id, ee_link_index)[12].decode()}, Position: {ee_pos}")

# Set camera
p.resetDebugVisualizerCamera(
    cameraDistance=4.0,
    cameraYaw=50,
    cameraPitch=-30,
    cameraTargetPosition=[2, 0, 1.0]
)

# Define initial and goal poses
initial_position = [0.0, 0.0, 1.5]
initial_orientation = p.getQuaternionFromEuler([0.0, np.pi, 0.0])  # EE pointing down
goal_position = [2.0, 0.5, 1.5]
goal_rpy = [0.0, np.pi, 0.0]  # Roll, Pitch, Yaw (EE pointing down)
goal_orientation = p.getQuaternionFromEuler(goal_rpy)

# Add sliders for goal position and orientation
x_slider = p.addUserDebugParameter("Goal X", -5.0, 10.0, goal_position[0])
y_slider = p.addUserDebugParameter("Goal Y", -5.0, 5.0, goal_position[1])
z_slider = p.addUserDebugParameter("Goal Z", 0.0, 5.0, goal_position[2])
roll_slider = p.addUserDebugParameter("Goal Roll", -np.pi, np.pi, goal_rpy[0])
pitch_slider = p.addUserDebugParameter("Goal Pitch", -np.pi, np.pi, goal_rpy[1])
yaw_slider = p.addUserDebugParameter("Goal Yaw", -np.pi, np.pi, goal_rpy[2])
execute_button = p.addUserDebugParameter("Execute IK", 0, 1, 0)  # Toggle slider

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
goal_rot_matrix = p.getMatrixFromQuaternion(goal_orientation)
goal_x_axis_vec = [goal_rot_matrix[0], goal_rot_matrix[3], goal_rot_matrix[6]]
goal_y_axis_vec = [goal_rot_matrix[1], goal_rot_matrix[4], goal_rot_matrix[7]]
goal_z_axis_vec = [goal_rot_matrix[2], goal_rot_matrix[5], goal_rot_matrix[8]]
goal_x_line = p.addUserDebugLine(goal_position,
                                 [goal_position[i] + target_axis_length * goal_x_axis_vec[i] for i in range(3)],
                                 [1, 0, 0], lineWidth=target_axis_width, lifeTime=0)
goal_y_line = p.addUserDebugLine(goal_position,
                                 [goal_position[i] + target_axis_length * goal_y_axis_vec[i] for i in range(3)],
                                 [0, 1, 0], lineWidth=target_axis_width, lifeTime=0)
goal_z_line = p.addUserDebugLine(goal_position,
                                 [goal_position[i] + target_axis_length * goal_z_axis_vec[i] for i in range(3)],
                                 [0, 0, 1], lineWidth=target_axis_width, lifeTime=0)

# Initialize EE axes
ee_x_line = None
ee_y_line = None
ee_z_line = None

print(f"Initial Pose (X,Y,Z): {initial_position}, RPY: [0.0, {np.pi:.2f}, 0.0]")
print(f"Initial Goal Pose (X,Y,Z): {goal_position}, RPY: [0.0, {np.pi:.2f}, 0.0]")

# Set initial pose
initial_ik_solution = p.calculateInverseKinematics(
    bodyUniqueId=robot_id,
    endEffectorLinkIndex=ee_link_index,
    targetPosition=initial_position,
    targetOrientation=initial_orientation,
    maxNumIterations=100,
    residualThreshold=1e-5
)
if len(initial_ik_solution) < len(movable_joints):
    print(f"Warning: Initial IK solution length ({len(initial_ik_solution)}) is less than movable joints ({len(movable_joints)}).")
    p.disconnect()
    exit()

initial_joint_positions = initial_ik_solution[:len(movable_joints)]
for i, joint_index in enumerate(movable_joints):
    p.resetJointState(robot_id, joint_index, initial_joint_positions[i])

# Lock initial pose
for i, joint_index in enumerate(movable_joints):
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=initial_joint_positions[i],
        force=2000,
        positionGain=0.05,
        velocityGain=1.5,
        maxVelocity=3.0
    )

print("\n--- Robot at Initial Pose ---")
for _ in range(480):  # Simulate for 2 seconds
    p.stepSimulation()
    time.sleep(1.0 / 480.0)

# Main loop
p.setRealTimeSimulation(1)
last_button_state = 0
last_goal_position = goal_position.copy()
last_goal_rpy = goal_rpy.copy()
while True:
    # Read slider values
    new_goal_x = p.readUserDebugParameter(x_slider)
    new_goal_y = p.readUserDebugParameter(y_slider)
    new_goal_z = p.readUserDebugParameter(z_slider)
    new_goal_roll = p.readUserDebugParameter(roll_slider)
    new_goal_pitch = p.readUserDebugParameter(pitch_slider)
    new_goal_yaw = p.readUserDebugParameter(yaw_slider)
    new_goal_position = [new_goal_x, new_goal_y, new_goal_z]
    new_goal_rpy = [new_goal_roll, new_goal_pitch, new_goal_yaw]

    # Update goal visualization if significant change
    if (any(abs(new_goal_position[i] - last_goal_position[i]) > 0.02 for i in range(3)) or
        any(abs(new_goal_rpy[i] - last_goal_rpy[i]) > 0.02 for i in range(3))):
        last_goal_position = new_goal_position.copy()
        last_goal_rpy = new_goal_rpy.copy()
        goal_orientation = p.getQuaternionFromEuler(last_goal_rpy)
        goal_rot_matrix = p.getMatrixFromQuaternion(goal_orientation)
        goal_x_axis_vec = [goal_rot_matrix[0], goal_rot_matrix[3], goal_rot_matrix[6]]
        goal_y_axis_vec = [goal_rot_matrix[1], goal_rot_matrix[4], goal_rot_matrix[7]]
        goal_z_axis_vec = [goal_rot_matrix[2], goal_rot_matrix[5], goal_rot_matrix[8]]
        print(f"Updated Goal Pose (X,Y,Z): {last_goal_position}, RPY: [{last_goal_rpy[0]:.2f}, {last_goal_rpy[1]:.2f}, {last_goal_rpy[2]:.2f}]")
        # Remove old goal debug lines
        p.removeUserDebugItem(goal_x_line)
        p.removeUserDebugItem(goal_y_line)
        p.removeUserDebugItem(goal_z_line)
        # Draw new goal debug lines
        goal_x_line = p.addUserDebugLine(last_goal_position,
                                        [last_goal_position[i] + target_axis_length * goal_x_axis_vec[i] for i in range(3)],
                                        [1, 0, 0], lineWidth=target_axis_width, lifeTime=0)
        goal_y_line = p.addUserDebugLine(last_goal_position,
                                        [last_goal_position[i] + target_axis_length * goal_y_axis_vec[i] for i in range(3)],
                                        [0, 1, 0], lineWidth=target_axis_width, lifeTime=0)
        goal_z_line = p.addUserDebugLine(last_goal_position,
                                        [last_goal_position[i] + target_axis_length * goal_z_axis_vec[i] for i in range(3)],
                                        [0, 0, 1], lineWidth=target_axis_width, lifeTime=0)

    # Check execute button
    button_state = p.readUserDebugParameter(execute_button)
    if button_state > last_button_state:  # Button pressed
        print("--- Executing IK and Moving Robot ---")
        # Remove previous EE axes if they exist
        if ee_x_line is not None:
            p.removeUserDebugItem(ee_x_line)
            p.removeUserDebugItem(ee_y_line)
            p.removeUserDebugItem(ee_z_line)
            ee_x_line = None
            ee_y_line = None
            ee_z_line = None

        # Step 1: Move linear axis
        if linear_axis_joint is not None:
            print("--- Moving Linear Axis ---")
            linear_target_position = last_goal_position[0]  # Assumes x-axis
            p.resetJointState(robot_id, linear_axis_joint, linear_target_position)
            # Lock other joints
            for i, joint_index in enumerate(movable_joints):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=initial_joint_positions[i],
                    force=2000,
                    positionGain=0.05,
                    velocityGain=1.5,
                    maxVelocity=3.0
                )
            for _ in range(480):  # Simulate for 2 seconds
                p.stepSimulation()
                time.sleep(1.0 / 480.0)

        # Step 2: Update ABB joints
        print("--- Moving ABB Joints to Goal Pose ---")
        # goal_ik_solution = p.calculateInverseKinematics(
        goal_ik_solution = calculateInverseKinematics(
            bodyUniqueId=robot_id,
            endEffectorLinkIndex=ee_link_index,
            targetPosition=last_goal_position,
            targetOrientation=goal_orientation,
            maxNumIterations=100,
            residualThreshold=1e-5
        )
        if len(goal_ik_solution) < len(movable_joints):
            print(f"Warning: Goal IK solution length ({len(goal_ik_solution)}) is less than movable joints ({len(movable_joints)}).")
            # Reset button
            p.removeUserDebugItem(execute_button)
            execute_button = p.addUserDebugParameter("Execute IK", 0, 1, 0)
            last_button_state = 0
            continue

        goal_joint_positions = goal_ik_solution[:len(movable_joints)]
        for i, joint_index in enumerate(abb_joints):
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
        # Keep linear axis locked
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=linear_axis_joint,
            controlMode=p.POSITION_CONTROL,
            targetPosition=linear_target_position,
            force=2000,
            positionGain=0.05,
            velocityGain=1.5,
            maxVelocity=3.0
        )
        for _ in range(480):  # Simulate for 2 seconds
            p.stepSimulation()
            time.sleep(1.0 / 480.0)

        # Lock final pose
        print("--- Robot Locked at Goal Pose ---")
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
        for _ in range(240):  # Simulate for 1 second to stabilize
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        # Visualize EE axes after locking pose
        ee_state = p.getLinkState(robot_id, ee_link_index)
        ee_pos = ee_state[0]
        ee_orient = ee_state[1]
        ee_rot_matrix = p.getMatrixFromQuaternion(ee_orient)
        ee_x_axis_vec = [ee_rot_matrix[0], ee_rot_matrix[3], ee_rot_matrix[6]]
        ee_y_axis_vec = [ee_rot_matrix[1], ee_rot_matrix[4], ee_rot_matrix[7]]
        ee_z_axis_vec = [ee_rot_matrix[2], ee_rot_matrix[5], ee_rot_matrix[8]]
        ee_x_line = p.addUserDebugLine(ee_pos,
                                       [ee_pos[i] + target_axis_length * ee_x_axis_vec[i] for i in range(3)],
                                       [1, 0, 1], lineWidth=target_axis_width, lifeTime=0)  # Magenta
        ee_y_line = p.addUserDebugLine(ee_pos,
                                       [ee_pos[i] + target_axis_length * ee_y_axis_vec[i] for i in range(3)],
                                       [1, 1, 0], lineWidth=target_axis_width, lifeTime=0)  # Yellow
        ee_z_line = p.addUserDebugLine(ee_pos,
                                       [ee_pos[i] + target_axis_length * ee_z_axis_vec[i] for i in range(3)],
                                       [0, 1, 1], lineWidth=target_axis_width, lifeTime=0)  # Cyan
        ee_rpy = p.getEulerFromQuaternion(ee_orient)
        print(f"EE Pose (X,Y,Z): {ee_pos}, RPY: [{ee_rpy[0]:.2f}, {ee_rpy[1]:.2f}, {ee_rpy[2]:.2f}]")
        print("--- EE Axes Drawn (Magenta=X, Yellow=Y, Cyan=Z) ---")

        # Reset button
        p.removeUserDebugItem(execute_button)
        execute_button = p.addUserDebugParameter("Execute IK", 0, 1, 0)
        last_button_state = 0

    last_button_state = button_state
    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()
print("Simulation ended.")