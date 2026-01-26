"""
3D Math utilities for parafoil dynamics.

Quaternion convention: [w, x, y, z] (scalar-first)
Rotation matrix C_IB: transforms vectors from Body frame to Inertial frame
    v_I = C_IB @ v_B
"""

import numpy as np
from typing import Tuple


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit length.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Normalized quaternion
    """
    norm = np.linalg.norm(q)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions: q_result = q1 * q2
    
    Convention: [w, x, y, z]
    
    Args:
        q1: First quaternion
        q2: Second quaternion
        
    Returns:
        Product quaternion
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate of a quaternion.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Conjugate quaternion [w, -x, -y, -z]
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_inverse(q: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a quaternion.
    For unit quaternions, this equals the conjugate.
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Inverse quaternion
    """
    return quat_conjugate(q) / np.dot(q, q)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    The rotation matrix C_IB transforms vectors from B to I:
        v_I = C_IB @ v_B
    
    Args:
        q: Quaternion [w, x, y, z] representing B->I rotation
        
    Returns:
        3x3 rotation matrix C_IB
    """
    q = normalize_quaternion(q)
    w, x, y, z = q
    
    # Precompute products
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    
    return np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy)],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx)],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]
    ])


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion using Shepperd's method.
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    
    # Ensure positive w for consistency
    if w < 0:
        q = -q
    
    return normalize_quaternion(q)


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create quaternion from Euler angles (ZYX convention).
    
    Args:
        roll: Rotation about x-axis [rad]
        pitch: Rotation about y-axis [rad]
        yaw: Rotation about z-axis [rad]
        
    Returns:
        Quaternion [w, x, y, z]
    """
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quat_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles from quaternion (ZYX convention).
    
    Args:
        q: Quaternion [w, x, y, z]
        
    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    q = normalize_quaternion(q)
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)  # Clamp for numerical stability
    pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quat_rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion: v_I = q * v_B * q^{-1}
    
    Args:
        q: Quaternion [w, x, y, z]
        v: Vector in body frame
        
    Returns:
        Rotated vector in inertial frame
    """
    # Convert vector to pure quaternion
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    
    # Apply rotation: q * v * q^{-1}
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    
    return result[1:4]


def quat_derivative(q: np.ndarray, w_B: np.ndarray) -> np.ndarray:
    """
    Compute quaternion time derivative given body angular velocity.
    
    q_dot = 0.5 * q ⊗ [0, w_B]
    
    Args:
        q: Current quaternion [w, x, y, z]
        w_B: Angular velocity in body frame [rad/s]
        
    Returns:
        Quaternion derivative
    """
    omega_quat = np.array([0.0, w_B[0], w_B[1], w_B[2]])
    return 0.5 * quat_multiply(q, omega_quat)


def skew(v: np.ndarray) -> np.ndarray:
    """
    Create skew-symmetric matrix from vector (for cross product).
    
    skew(v) @ u = v × u
    
    Args:
        v: 3D vector
        
    Returns:
        3x3 skew-symmetric matrix
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Cross product of two 3D vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cross product a × b
    """
    return np.cross(a, b)


def angle_wrap(angle: float) -> float:
    """
    Wrap angle to [-pi, pi].
    
    Args:
        angle: Angle in radians
        
    Returns:
        Wrapped angle
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def safe_normalize(v: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Safely normalize a vector, returning zero vector if input is too small.
    
    Args:
        v: Vector to normalize
        eps: Minimum norm threshold
        
    Returns:
        Normalized vector or zero vector
    """
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def compute_aero_angles(v_B: np.ndarray, eps: float = 1e-6) -> Tuple[float, float, float]:
    """
    Compute aerodynamic angles from body-frame velocity.
    
    Args:
        v_B: Velocity in body frame [u, v, w] (forward, right, down)
        eps: Small value for numerical protection
        
    Returns:
        Tuple of (V, alpha, beta) where:
            V: Airspeed [m/s]
            alpha: Angle of attack [rad]
            beta: Sideslip angle [rad]
    """
    u, v, w = v_B
    
    # Total airspeed with protection
    V = np.sqrt(u*u + v*v + w*w)
    V_safe = max(V, eps)
    
    # Angle of attack (in x-z plane)
    # alpha = arctan(w/u)
    u_safe = u if abs(u) > eps else eps * np.sign(u) if u != 0 else eps
    alpha = np.arctan2(w, u_safe)
    
    # Sideslip angle
    # beta = arcsin(v/V)
    sin_beta = np.clip(v / V_safe, -1.0, 1.0)
    beta = np.arcsin(sin_beta)
    
    return V, alpha, beta
