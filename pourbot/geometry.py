"""Pose maths in Viam's conventions: millimetres, orientation vectors, degrees."""
import math

import numpy as np


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(rad):
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ov_to_matrix(ox, oy, oz, theta_deg):
    v = np.array([ox, oy, oz], dtype=float)
    v /= np.linalg.norm(v)
    lon = math.atan2(v[1], v[0]) if 1 - abs(v[2]) > 1e-4 else 0.0
    lat = math.acos(max(-1.0, min(1.0, v[2])))
    return rot_z(lon) @ rot_y(math.degrees(lat)) @ rot_z(math.radians(theta_deg))


def matrix_to_ov(r):
    """Inverse of ov_to_matrix. Returns (ox, oy, oz, theta_deg)."""
    z = r[:, 2]
    lon = math.atan2(z[1], z[0]) if 1 - abs(z[2]) > 1e-4 else 0.0
    lat = math.acos(max(-1.0, min(1.0, z[2])))
    rest = rot_y(-math.degrees(lat)) @ rot_z(-lon) @ r
    return float(z[0]), float(z[1]), float(z[2]), math.degrees(math.atan2(rest[1, 0], rest[0, 0]))


def pose_to_tf(p):
    """p: any object or dict with x,y,z,o_x,o_y,o_z,theta. Returns (R, t)."""
    g = p.get if isinstance(p, dict) else lambda k, d=0.0: getattr(p, k, d)
    return (ov_to_matrix(g("o_x", 0.0), g("o_y", 0.0), g("o_z", 1.0), g("theta", 0.0)),
            np.array([g("x", 0.0), g("y", 0.0), g("z", 0.0)], dtype=float))


def tf_to_pose(r, t):
    ox, oy, oz, th = matrix_to_ov(r)
    return dict(x=float(t[0]), y=float(t[1]), z=float(t[2]), o_x=ox, o_y=oy, o_z=oz, theta=th)
