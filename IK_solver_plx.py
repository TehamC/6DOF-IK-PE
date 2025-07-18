import pybullet as p
import numpy as np

def invert_quaternion(q):
    q = np.array(q, dtype=np.float64)
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    return q_conj / np.dot(q, q)

def quaternion_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    return [x, y, z, w]

def normalize_angle(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi

def calculateInverseKinematics(
    bodyUniqueId,
    endEffectorLinkIndex,
    targetPosition,
    targetOrientation=None,
    maxNumIterations=100,
    residualThreshold=1e-4,
    orientation_weight=1.0
):
    """Custom numerical IK for arbitrary chain (rail+arm)"""
    target_pos = np.array(targetPosition, dtype=np.float64)
    target_ori = np.array(targetOrientation, dtype=np.float64) if targetOrientation is not None else None

    num_joints = p.getNumJoints(bodyUniqueId)
    joint_indices = [
        i for i in range(num_joints)
        if p.getJointInfo(bodyUniqueId, i)[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]
    ]
    q = np.array([p.getJointState(bodyUniqueId, i)[0] for i in joint_indices], dtype=np.float64)

    for iteration in range(maxNumIterations):
        for i, idx in enumerate(joint_indices):
            p.resetJointState(bodyUniqueId, idx, q[i])

        link_state = p.getLinkState(bodyUniqueId, endEffectorLinkIndex, computeForwardKinematics=True)
        current_pos = np.array(link_state[4])
        current_ori = np.array(link_state[5])

        pos_error = target_pos - current_pos

        if target_ori is not None:
            inv_current_ori = invert_quaternion(current_ori)
            diff_quat = quaternion_multiply(inv_current_ori, target_ori)
            ori_error = 0.5 * orientation_weight * np.array(diff_quat[:3])
            error = np.concatenate((pos_error, ori_error))
        else:
            error = pos_error

        if np.linalg.norm(error) < residualThreshold:
            break

        delta = 1e-6
        dof = len(joint_indices)
        J = np.zeros((6 if target_ori is not None else 3, dof))

        for j in range(dof):
            original = q[j]

            q[j] = original + delta
            for k, idx in enumerate(joint_indices):
                p.resetJointState(bodyUniqueId, idx, q[k])
            plus_state = p.getLinkState(bodyUniqueId, endEffectorLinkIndex, computeForwardKinematics=True)
            pos_plus = np.array(plus_state[4])
            ori_plus = np.array(plus_state[5])

            q[j] = original - delta
            for k, idx in enumerate(joint_indices):
                p.resetJointState(bodyUniqueId, idx, q[k])
            minus_state = p.getLinkState(bodyUniqueId, endEffectorLinkIndex, computeForwardKinematics=True)
            pos_minus = np.array(minus_state[4])
            ori_minus = np.array(minus_state[5])

            q[j] = original  # Restore

            J_pos = (pos_plus - pos_minus) / (2 * delta)
            if target_ori is not None:
                inv_ori_minus = invert_quaternion(ori_minus)
                delta_q = quaternion_multiply(inv_ori_minus, ori_plus)
                J_ori = 0.5 * orientation_weight * np.array(delta_q[:3]) / (2 * delta)
                J[:, j] = np.concatenate((J_pos, J_ori))
            else:
                J[:, j] = J_pos

        # Damped pseudo-inverse step (to avoid singularities)
        JT = J.T
        lambda_diag = 1e-6 * np.eye(J.shape[0])
        dq = np.dot(JT, np.linalg.solve(np.dot(J, JT) + lambda_diag, error))

        dq = np.clip(dq, -0.05, 0.05)
        q += dq
        # Normalize angles for revolute joints, skip prismatic axis
        for i, idx in enumerate(joint_indices):
            joint_type = p.getJointInfo(bodyUniqueId, idx)[2]
            if joint_type == p.JOINT_REVOLUTE:
                q[i] = normalize_angle(q[i])

    return q.tolist()
