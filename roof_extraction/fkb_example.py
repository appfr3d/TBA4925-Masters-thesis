# Tutorial on geopandas here
# https://folk.ntnu.no/sverrsti/TVB4105-H2020/geopandas_intro.html

import os
import pandas as pd
import geopandas as gpd
import fiona as fi
import matplotlib.pyplot as plt


# FKB-Bygning data downloaded from geonorge.no
# Trondheim is about 20.1MB, while Trondelag is about 108.5MB! So use Trondheim as it is much faster to load!
file_name = "Basisdata_5001_Trondheim_5972_FKB-Bygning_FGDB.gdb"
current_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(current_folder)
building_path = os.path.join(project_folder, "data/building_footprints", file_name)


# Which maplayers exist within the dataset?
print(fi.listlayers(building_path))


# Extract one maplayer
print('Loading FKB-Bygning data...')

omrade = gpd.read_file(building_path, layer = 'fkb_bygning_omrade')
print("done!")

# Which attributes (columns) is in this maplayer?
# omrade.info()

# How does the first five rows look like?
# print(omrade.head())


large_buildings = omrade[omrade.SHAPE_Area > 30]
large_buildings.info()
print(large_buildings.head())

# building_zero = omrade[omrade.index == 0]

# building_zero["geometry"].plot()



# How does the data look like on a map? Coloring is done with the column value.
# print('Plotting fkb_bygning_omrade...')

# omrade.plot(column="bygningsnummer")
# print("done!")

# plt.show()


# Crop point cloud:
# http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html#Crop-point-cloud
# demo_crop_data = o3d.data.DemoCropPointCloud()
# pcd = o3d.io.read_point_cloud(demo_crop_data.point_cloud_path)
# vol = o3d.visualization.read_selection_polygon_volume(demo_crop_data.cropped_json_path)
# chair = vol.crop_point_cloud(pcd)
