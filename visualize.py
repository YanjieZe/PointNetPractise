import torch
import numpy as np
import open3d as o3d
from pointnet import STN3d, STNkd, feature_transform_reguliarzer

def visualize(data=None):
    print("visualizing............")

    if data.any():
        pass
    else:
        data = np.load("./hw/0.npy")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    visualize()