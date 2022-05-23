import numpy as np
import open3d as o3d

from features.helpers import read_roof_cloud, get_roof_folder, normalize_cloud

file_name = "32-1-510-215-53-test-1.ply"
cloud = read_roof_cloud(file_name)
cloud = normalize_cloud(cloud)


# print(cloud.normals)
# if np.asarray(cloud.normals).shape[0] == 0:
#   print('no normals')

# Add normals to cloud
cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=80))
cloud.orient_normals_consistent_tangent_plane(30)

feature = o3d.pipelines.registration.compute_fpfh_feature(cloud, o3d.geometry.KDTreeSearchParamKNN(knn=10))


print(feature.data.shape)