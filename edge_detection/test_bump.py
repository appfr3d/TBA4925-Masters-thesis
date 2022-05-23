import os
import numpy as np
import open3d as o3d
from matplotlib import cm

from features.helpers import read_roof_cloud, save_scaled_feature_image, normalize_cloud, get_roof_folder
from features.feature import ScalableFeature, ScalableFeatureState
from features.covariance_eigenvalue import CovarianceEigenvalue
from features.normal_cluster import NormalCluster

# (CovarianceEigenvalue, ScalableFeatureState, "covariance_eigenvalue"), 
# feature_class(feature_state_class(pcd))

print('Reading cloud...')
file_name = "104__thin_ground_points_121.ply"
cloud = read_roof_cloud(file_name)
print(cloud)
print('Normalizing cloud...')
cloud = normalize_cloud(cloud)


# print(cloud.normals)
# if np.asarray(cloud.normals).shape[0] == 0:
#   print('no normals')
feature_types = ["covariance_eigenvalue", "normal_cluster"]
feature_type = feature_types[0]

print('Estimating normals...')
# Add normals to cloud
cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=80))
cloud.orient_normals_consistent_tangent_plane(30)

o3d.visualization.draw_geometries([cloud])

print('Creating ScalableFeatureState...')
state = ScalableFeatureState(cloud)
if feature_type == "covariance_eigenvalue":
  f = CovarianceEigenvalue(state)
else:
  f = NormalCluster(state)

print('Running calculations...')
all_labels = f.run()


colormap = cm.get_cmap('rainbow') 
colors = np.array([colormap(l) for l in all_labels[4]])
cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([cloud])


vis = o3d.visualization.Visualizer()
vis.create_window(width=1000, height=1000)
vis.add_geometry(cloud)

print('Displaying result for each scale...')
colormap = cm.get_cmap('rainbow') 

for label_i in range(all_labels.shape[0]):
  # Remove treshold for visualization
  labels = all_labels[label_i]
  save_scaled_feature_image(vis, cloud, labels, os.path.join(get_roof_folder(), "bumps", feature_type) , str(label_i))
