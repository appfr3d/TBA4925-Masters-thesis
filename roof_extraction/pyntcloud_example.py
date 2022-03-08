import os
from pyntcloud import PyntCloud

cloud_file_name_base = "32-1-510-215-53"
cloud_file_name = cloud_file_name_base + ".ply"

current_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(current_folder)
data_folder = os.path.join(project_folder, "data/point_clouds/raw")
cloud_path = os.path.join(data_folder, cloud_file_name)

cloud = PyntCloud.from_file(cloud_path)

attrs = [attr for attr in dir(cloud) if not attr.startswith('__')]
for a in attrs:
  print(a)
