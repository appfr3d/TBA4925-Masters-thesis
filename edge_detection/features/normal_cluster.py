import numpy as np
import open3d as o3d
import copy
from feature import Feature


# TODO: Make it a scaled feature. Will work much better!

class NormalCluster(Feature):
  def run(self):
    normal_cloud = copy.deepcopy(self.cloud)

    # Add normals to cloud
    normal_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=80))
    normal_cloud.orient_normals_consistent_tangent_plane(30)

    o3d.visualization.draw_geometries([normal_cloud])

    # Set the points as the unit normals
    normal_cloud.points = normal_cloud.normals

    # Uncomment to visualize normals as points on a sphere
    o3d.visualization.draw_geometries([normal_cloud])

    labels = np.array(normal_cloud.cluster_dbscan(eps=0.1, min_points=100))

    return labels

if __name__ == "__main__":
  from helpers import read_roof_cloud, write_roof_cloud_result

  file_name = "32-1-510-215-53-roof-2-shift.ply"
  cloud = read_roof_cloud(file_name)

  f = NormalCluster(cloud)

  labels = f.run()

  colors = np.zeros((labels.shape[0], 3))
  colors += [0.6, 0.6, 0.6]
  
  colors[labels < 0] = [0, 1, 0] # Color noise from dbscan cluster as green

  cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

  o3d.visualization.draw_geometries([cloud])
  file_out_name = "32-1-510-215-53-normal_cluster-shift.ply"
  write_roof_cloud_result(file_name, cloud)


