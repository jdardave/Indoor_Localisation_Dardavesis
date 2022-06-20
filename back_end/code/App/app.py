from cv2 import resize, threshold
from matplotlib import colors
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import glob 
import time
import copy
import sys
from flask import Flask,render_template,request,redirect,url_for, session
from werkzeug.utils import secure_filename
import os
from arcgis.gis import GIS
from arcgis.features import FeatureLayerCollection
from arcgis.features import FeatureLayer
import uuid
from difflib import SequenceMatcher

id = uuid.uuid1()
# CHANGE PATH
UPLOAD_FOLDER = 'C:/Users/jdard/OneDrive/Desktop/Thesis/back_end/data'
ALLOWED_EXTENSIONS = {'ply'}
result=[]
names_adjacent=[]
pc_db_new=[]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def upload_file():
   return render_template('upload.html')
#    return render_template('html_map.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def uploader():
   if request.method == 'POST':
      f = request.files['file']
      f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
    #   print("file saved")
      return render_template("result.html",room_name=main1(f.filename))
      # return 'succesful operation, waiting for localisation...'
      # return redirect(url_for('upload_file'))

def preparation (pc,voxel_size):
    """"Downsampling, normals,fpfh,inliers"""

    radius_normal = voxel_size * 1.5
    # print(":: Load two point clouds and disturb initial pose.")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # pcd_user.transform(trans_init)
    voxel_down_pcd = pc.voxel_down_sample(voxel_size)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=30,
                                                    std_ratio=1)
    # display_inlier_outlier(voxel_down_pcd, ind)

    plane_model, inliers = cl.segment_plane(distance_threshold=0.4,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model

    inlier_cloud = cl.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    inlier_cloud.colors=cl.colors
    outlier_cloud = cl.select_by_index(inliers, invert=True)
    outlier_cloud.colors=cl.colors
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    # outlier_cloud.estimate_normals(
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    radius_feature = voxel_size * 3

    pc_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            outlier_cloud,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pc,outlier_cloud,pc_fpfh


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


#Vanilla ICP -BIGGER DISTANCE MIGHT IMPROVE ALIGNMENT (EXPERIMENT BETWEEN 2-3)
# def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
#     distance_threshold = voxel_size * 3
#     result = o3d.pipelines.registration.registration_icp(
#         source, target, distance_threshold,result_ransac.transformation ,
#         o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
#     return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    # distance_threshold = voxel_size * 5
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
        source_down, target_down, radius, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=iter))
    return result

def main1(file_name):
    start = time.time()
    routes_url="https://services3.arcgis.com/jR9a3QtlDyTstZiO/arcgis/rest/services/Routes_adjacent/FeatureServer/29"
    layer = FeatureLayer(routes_url)
    # demo_crop_geometry()
    # demo_crop_geometry(pcd_user)
    # o3d.visualization.draw_geometries([pcd_user])
    voxel_size=0.15
    # pcd_user=crop_user_pc(pcd_user)
    # o3d.visualization.draw_geometries([pcd_user])
    pcd_user = o3d.io.read_point_cloud(UPLOAD_FOLDER +'/' + file_name)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # pcd_user.transform(trans_init)
    # draw_registration_result(pcd_user,pcd_user, np.identity(4))
    pcd_user.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    pcd_user, outlier_user, source_fpfh = preparation(pcd_user, voxel_size)
    global result_ransac
    path = "../back_end/data/database/"
    all_files = glob.glob(path + "*.ply")
#split string with delimeter /
    fitness={}
    if len(result)==0:
        for pc_db in all_files:
            # string = "08.02.00.470"
            pcd_db = o3d.io.read_point_cloud("{}".format(pc_db))
            # draw_registration_result(pcd_user,pcd_db, np.identity(4))
            pcd_db.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            pcd_db, outlier_db, target_fpfh = preparation(pcd_db, voxel_size)
            result_ransac=execute_global_registration(outlier_user, outlier_db,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
                # print(result_ransac)
                # draw_registration_result(outlier_user, outlier_db, result_ransac.transformation)
            result_icp = refine_registration(pcd_user, pcd_db, source_fpfh, target_fpfh,
                                            voxel_size)
                # print(result_icp)
                # draw_registration_result(pcd_user, pcd_db, result_icp.transformation)
                # print("Inlier Fitness of {}: {:.3f}".format(pc_db,result_icp.fitness))
                # print("Inlier RMSE: {:.3f}".format(result_icp.inlier_rmse))
            pc_db=pc_db.split('\\')
            pc_db_1=pc_db[1].split('-')
                # print(pc_db_1)
            fitness[str(pc_db_1[0])]=result_icp.fitness
            max_key=max(fitness,key=fitness.get)
    elif len(result)>=1:
        for i in range(len(result)):
            names_adjacent.clear()
            pc_db_new.clear()
            query_result1 = layer.query(where ="FirstStopName='{}'".format(result[i]),out_fields='LastStopName')
            for i in range(len(query_result1.features)):
                names_adjacent.append(query_result1.features[i].as_row[0][1])
            for name in names_adjacent:
                for pc_db in all_files:
                    ratio=SequenceMatcher(None, name, pc_db).ratio()
                    # print(ratio)
                    if ratio>0.2:
                        pc_db_new.append(pc_db)
        print(pc_db_new)
            
        for pc_db in pc_db_new:
            pcd_db = o3d.io.read_point_cloud("{}".format(pc_db))
                    # draw_registration_result(pcd_user,pcd_db, np.identity(4))
            pcd_db.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
            pcd_db, outlier_db, target_fpfh = preparation(pcd_db, voxel_size)
            result_ransac=execute_global_registration(outlier_user, outlier_db,
                                                            source_fpfh, target_fpfh,
                                                            voxel_size)
                    # print(result_ransac)
                    # draw_registration_result(outlier_user, outlier_db, result_ransac.transformation)
            result_icp = refine_registration(pcd_user, pcd_db, source_fpfh, target_fpfh,
                                                voxel_size)
                    # print(result_icp)
                    # draw_registration_result(pcd_user, pcd_db, result_icp.transformation)
                    # print("Inlier Fitness of {}: {:.3f}".format(pc_db,result_icp.fitness))
                    # print("Inlier RMSE: {:.3f}".format(result_icp.inlier_rmse))
            pc_db=pc_db.split('\\')
            pc_db_1=pc_db[1].split('-')
                    # print(pc_db_1)
            fitness[str(pc_db_1[0])]=result_icp.fitness
            max_key=max(fitness,key=fitness.get)        
        # print(pc_db_new)
    long_string= "08.02.00." + max_key
    result.append(long_string)
    print(result)
    
    # counter=

    # for i in range(len(result)-1):
    #     if len(result)>1:
    #         query_result=layer.query(where = "FirstStopName='{}' and LastStopName='{}'".format(result[i],result[i+1]))
    #         counter=query_result.features[0].get_value(field_name ="RouteUsed")
    #         layer.calculate(where = "FirstStopName='{}' and LastStopName='{}'".format(result[i],result[i+1]), calc_expression = [{"field": "RouteUsed","value": "{}".format(counter+1)},{"field": "User_ID","value": "{}".format(id)}],
    #         return_edit_moment=True,sessionid=id,future=True)


    if len(result)>1:
            query_result=layer.query(where = "FirstStopName='{}' and LastStopName='{}'".format(result[-2],result[-1]))
            counter=query_result.features[0].get_value(field_name ="RouteUsed")
            layer.calculate(where = "FirstStopName='{}' and LastStopName='{}'".format(result[-2],result[-1]), calc_expression = [{"field": "RouteUsed","value": "{}".format(counter+1)},{"field": "User_ID","value": "{}".format(id)}],
            return_edit_moment=True,sessionid=id,future=True)
            # print(counter+1)
            # layer.append()
    # counter+=1
        # fitness.append(result_icp.fitness)

    # print(fitness)
    test = os.listdir(UPLOAD_FOLDER)
    for file in test:
        if file.endswith(".ply"):
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    print("Time elapsed: {:.4f}".format(time.time()-start))
    return long_string
    


    
if __name__ == '__main__':
   app.run(debug = False)



   


