import numpy as np
import open3d as o3d
import trimesh
import matplotlib.pyplot as plt
from matplotlib import cm
import copy

from features.helpers import normalize_cloud

# pcd = o3d.io.read_point_cloud("../data/point_clouds/raw/32-1-510-215-53.ply")
# o3d.visualization.draw_geometries([pcd], width=1000, height=1000)

cloud_file_name = "../data/point_clouds/classified_roofs/32-1-510-215-53-test-1.ply"
tri_cloud = trimesh.load(cloud_file_name)
labels = tri_cloud.metadata["ply_raw"]["vertex"]["data"]["scalar_Classification"].astype(int)
cloud = o3d.io.read_point_cloud(cloud_file_name)

# Visualize GT edges
colors = np.zeros((labels.shape[0], 3))
colors += [0.6, 0.6, 0.6]
colors[labels < 0.5] = [0, 1, 0] # Color positive values as green
cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
cloud.points = cloud.points
o3d.visualization.draw_geometries([cloud], width=1024, height=1024)



cloud, downsampling_factor = normalize_cloud(cloud)
points = np.asarray(cloud.points)



z = points[:, 2]
# minus = z[z<=0]
# plus = z[z>0]
# mean_minus = np.median(minus)
# mean_plus = np.median(plus)
# diff = np.max(z) + np.min(z)
# # diff = mean_plus - mean_minus 
# z -= diff/2
z_pow2 = np.power(z, 2)
# print('labels:', labels)
# print('z len:', z.shape[0])
# print('mean z', np.mean(z))
# print('z non:', z[labels==1])
# print('z non len:', z[labels==1].shape[0])
# print('z edg:', z[(1-labels)==1])
# print('z edg len:', z[(1-labels)==1].shape[0])

# non = z_pow2[labels]
plt.subplot(1, 2, 1)
plt.scatter(z[labels==1], z[labels==1], label="Non-edge", color="red")
plt.scatter(z[(1-labels)==1], z[(1-labels)==1], label="Edge", color="blue", marker='s', s=5)
plt.grid(True)
plt.legend()
plt.xlabel("Z")
plt.ylabel("Z")

plt.subplot(1, 2, 2)
plt.scatter(z[labels==1], z_pow2[labels==1], label="Non-edge", color="red")
plt.scatter(z[(1-labels)==1], z_pow2[(1-labels)==1], label="Edge", color="blue", marker='s', s=5)
plt.grid(True)
plt.legend()
plt.xlabel("Z")
plt.ylabel("Z^2")
plt.show()


'''
# Visualize for presentation
colors = np.zeros((labels.shape[0], 3))
o3d.visualization.draw_geometries([cloud], width=1024, height=1024)

c = o3d.geometry.PointCloud()
c.points = cloud.normals
c.normals = cloud.normals
center = o3d.geometry.PointCloud()
s = np.asarray(cloud.points).shape
center.points = o3d.utility.Vector3dVector(np.zeros(s))
center.normals = cloud.normals
center.colors = o3d.utility.Vector3dVector(np.zeros(s) + [1, 0, 0])
o3d.visualization.draw_geometries([c, center], width=1024, height=1024)

# Visualize dbscan
labels = np.array(c.cluster_dbscan(eps=0.05, min_points=np.floor_divide(np.asarray(cloud.points).shape[0], 5)))
clusters = np.unique(labels)
print("clusters:", clusters)
color_labels = np.divide(labels + 1, clusters.shape[0] - 1)
test_cloud = o3d.geometry.PointCloud()
test_cloud.points = c.points
colormap = cm.get_cmap('rainbow')
test_colors = np.array([colormap(l) for l in color_labels])
test_cloud.colors = o3d.utility.Vector3dVector(test_colors[:, :3])
o3d.visualization.draw_geometries([test_cloud], width=1024, height=1024)

# Visualize result
colors = np.zeros((labels.shape[0], 3))
colors += [0.6, 0.6, 0.6]
colors[labels < 0] = [0, 1, 0] # Color positive values as green
c.colors = o3d.utility.Vector3dVector(colors[:, :3])
c.points = cloud.points
o3d.visualization.draw_geometries([c], width=1024, height=1024)
'''
