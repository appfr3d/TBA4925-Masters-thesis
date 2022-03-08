import os
import laspy
import numpy as np

cloud_file_name_base = "32-1-510-215-53"
cloud_file_name = cloud_file_name_base + ".laz"

current_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(current_folder)
data_folder = os.path.join(project_folder, "data/point_clouds/raw")
cloud_path = os.path.join(data_folder, cloud_file_name)

cloud = laspy.read(cloud_path)

for dimension in cloud.point_format.dimensions:
    print(dimension.name)



classes = np.array(cloud.classification)

print(np.min(classes))

for i in range(1, np.max(classes) + 1):
  print('Class', i, ':', np.sum(classes == i))

