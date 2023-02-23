from typing import Sequence
from cv2 import threshold
from matplotlib import colors
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import glob 
import time
import copy
import sys
import os
from difflib import SequenceMatcher


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    print(o3d.geometry.PointCloud.get_center(source_temp))
    print(o3d.geometry.PointCloud.get_center(target))
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preparation (pc,voxel_size):
    """"Downsampling, normals,fpfh,inliers"""
    # print(len(np.asarray(pc.points)))
    radius_normal = voxel_size * 1.5
    # print(":: Load two point clouds and disturb initial pose.")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # pcd_user.transform(trans_init)
    # o3d.visualization.draw_geometries([pc])
    voxel_down_pcd = pc.voxel_down_sample(voxel_size)
    # print(len(np.asarray(voxel_down_pcd.points)))
    # o3d.visualization.draw_geometries([voxel_down_pcd])
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                    std_ratio=1)
    # display_inlier_outlier(voxel_down_pcd, ind)
    # print(len(np.asarray(cl.points)))
    plane_model, inliers = cl.segment_plane(distance_threshold=0.4,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    
    inlier_cloud = cl.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # inlier_cloud.colors=cl.colors
    outlier_cloud = cl.select_by_index(inliers, invert=True)
    # outlier_cloud.colors=cl.colors
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    # print(len(np.asarray(outlier_cloud.points)))
    outlier_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    radius_feature = voxel_size * 3

    pc_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            outlier_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pc,outlier_cloud,pc_fpfh

# GLOBAL REGISTRATION WITH RANSAC
def execute_global_registration(outlier_user, outlier_db, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        outlier_user, outlier_db, source_fpfh, target_fpfh, True, 
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)) 
    return result

# #FAST GLOBAL REGISTRATION
# def execute_global_registration(outlier_user, outlier_db, source_fpfh,
#                                 target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 1.5
#     result =  o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
#         outlier_user, outlier_db, source_fpfh, target_fpfh,
#          o3d.pipelines.registration.FastGlobalRegistrationOption(True,True,
#             maximum_correspondence_distance=distance_threshold))
#     return result


#Vanilla ICP -BIGGER DISTANCE MIGHT IMPROVE ALIGNMENT (EXPERIMENT BETWEEN 2-3)
#ICP
# def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 3
#     result = o3d.pipelines.registration.registration_icp(
#         source, target, distance_threshold,result_ransac.transformation ,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
#     return result


# Generalised ICP

# def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 3
#     result = o3d.pipelines.registration.registration_generalized_icp(
#         source, target, distance_threshold,result_ransac.transformation ,
#         o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
#     return result



# COLORED ICP
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3
    voxel_radius = [0.1, 0.15, 0.2]
    max_iter = [15, 30, 50]
    # current_transformation = np.identity(4)
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, radius,result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=iter))
    return result

# def demo_crop_geometry():
#     print("Demo for manual geometry cropping")
#     print(
#         "1) Press 'Y' twice to align geometry with negative direction of y-axis"
#     )
#     print("2) Press 'K' to lock screen and to switch to selection mode")
#     print("3) Drag for rectangle selection,")
#     print("   or use ctrl + left click for polygon selection")
#     print("4) Press 'C' to get a selected geometry and to save it")
#     print("5) Press 'F' to switch to freeview mode")
#     pcd_user = o3d.io.read_point_cloud("../back_end/data/user/450-30.ply")
#     o3d.visualization.draw_geometries_with_editing([pcd_user])


def main():
    start = time.time()
    # demo_crop_geometry()x
    # demo_crop_geometry(pcd_user)
    # o3d.visualization.draw_geometries([pcd_user])
    voxel_size=0.15
    # pcd_user=crop_user_pc(pcd_user)
    # o3d.visualization.draw_geometries([pcd_user])
    pcd_user = o3d.io.read_point_cloud("../back_end/data/user/808.ply") # In case the point clouds from pix4d are used this path changes to "../back_end/data/pix4d_user/"
    # print(len(np.asarray(pcd_user.points)))
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # pcd_user.transform(trans_init)
    # draw_registration_result(pcd_user,pcd_user, np.identity(4))
    pcd_user.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    pcd_user, outlier_user, source_fpfh = preparation(pcd_user, voxel_size)
    # print(len(np.asarray(outlier_user.points)))
    global result_ransac
    path = "../back_end/data/database/" # In case the point clouds from pix4d are used this path changes to "../back_end/data/pix4d_database/"
    all_files = glob.glob(path + "*.ply")
#split string with delimeter /
    fitness={}
    for pc_db in all_files:
        
        # string = "08.02.00.430"
        # print(SequenceMatcher(None, string, pc_db).ratio())

        pcd_db = o3d.io.read_point_cloud("{}".format(pc_db))
        # print(len(np.asarray(pcd_db.points)))
        # draw_registration_result(pcd_user,pcd_db, np.identity(4))
        pcd_db.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        pcd_db, outlier_db, target_fpfh = preparation(pcd_db, voxel_size)
        result_ransac=execute_global_registration(outlier_user, outlier_db,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
        # print(len(np.asarray(outlier_db.points)),pc_db)
        # print(result_ransac)
        pcd_user.paint_uniform_color([1, 0.706, 0])
        pcd_db.paint_uniform_color([0, 0.651, 0.929])
        # o3d.visualization.draw_geometries([pcd_user,pcd_db])

        result_icp = refine_registration(pcd_user, pcd_db, source_fpfh, target_fpfh,
                                    voxel_size)
        
        draw_registration_result(pcd_user, pcd_db, result_icp.transformation)

        print(result_icp)

        # print(o3d.geometry.PointCloud.get_center(pcd_user))
        # print(o3d.geometry.PointCloud.get_center(pcd_db))
        
        # evaluation = o3d.pipelines.registration.evaluate_registration(
        # source_down, target_down, voxel_size*3, result_icp.transformation)
        # print(evaluation)
        # print("Inlier Fitness of {}: {:.3f}".format(pc_db,result_icp.fitness))
        # print("Inlier RMSE: {:.3f}".format(result_icp.inlier_rmse))
        pc_db=pc_db.split('\\') # or pc_db = pc_db.split('/') for Mac users
        pc_db_1=pc_db[1].split('-')
        # print(pc_db_1)
        fitness[str(pc_db_1[0])]=result_icp.fitness
        max_key=max(fitness,key=fitness.get) 
        # fitness.append(result_icp.fitness)
    print("Time elapsed: {:.4f}".format(time.time()-start))
    # print(fitness)
    print(max_key)

if __name__ == "__main__":
    main()

