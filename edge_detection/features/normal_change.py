import open3d as o3d
import copy

class NormalChange():
  def __init__(self, cloud) -> None:
    self.cloud = copy.deepcopy(cloud)
  
  def run(self, scale: float):
    # Add normals to cloud
    self.cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=scale, max_nn=30))
    self.cloud.orient_normals_consistent_tangent_plane(30)
    
    print(scale)
