import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob 
import time


def orb(img_user,img_ref):
    orb = cv.ORB_create(nfeatures=5000)
    # Keypoints & Descriptors
    kp1, des1 = orb.detectAndCompute(img_user,None)
    kp2, des2 = orb.detectAndCompute(img_ref,None)
    img_user = cv.drawKeypoints(img_user, kp1, None, color=(0,255,0), flags=0)
    img_ref = cv.drawKeypoints(img_ref, kp2, None, color=(0,255,0), flags=0)

    return kp1,kp2,des1,des2

def sift(img_user,img_ref):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_user,None)
    kp2, des2 = sift.detectAndCompute(img_ref,None)
    return kp1,kp2,des1,des2


def brute_force(img_user,img_ref):
    # ORB
    bf = cv.BFMatcher(cv.NORM_HAMMING) 
    # SIFT
    # bf = cv.BFMatcher(cv.NORM_L2) 
    kp1,kp2,des1,des2=orb(img_user,img_ref)
    matches = bf.knnMatch(des1,des2,2)
    print(matches[0])
    ratio_thresh = 0.6
    passed_matches= []
    for i,j in matches:
        if i.distance < ratio_thresh * j.distance:
            passed_matches.append(i)
    # passed_matches=sorted(passed_matches, key = lambda x:x.distance)
    return passed_matches
    # matches = sorted(matches, key = lambda x:x.distance)

def flann(img_user,img_ref):
    bf = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    kp1,kp2,des1,des2=orb(img_user,img_ref)
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    matches= bf.knnMatch(des1,des2,2)
    ratio_thresh = 0.6
    passed_matches= []
    for i,j in matches:
        if i.distance < ratio_thresh * j.distance:
            passed_matches.append(i)
    return passed_matches



def main():
    start = time.time()
    path = "../back_end/data/single_image_database/"
    all_files = glob.glob(path + "*.jpg")
    img_user = cv.imread('../back_end/data/single_image_user/808_user_photo.jpg',cv.IMREAD_GRAYSCALE)
    for all_img_ref in all_files:
        img_ref = cv.imread("{}".format(all_img_ref),cv.IMREAD_GRAYSCALE)
        kp1,kp2,des1,des2 = orb(img_user,img_ref)
        # kp1,kp2,des1,des2 = sift(img_user,img_ref)
        # matches = brute_force(img_user,img_ref)
        matches_fl =flann(img_user,img_ref)
        # plt.imshow(img_user), plt.show()
        # plt.imshow(img_ref), plt.show()
        keypoints_1 = cv.drawKeypoints(img_user, kp1, 3, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # keypoints_2 = cv.drawKeypoints(img_ref, kp2, None, color=(0,255,0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img3 = cv.drawMatches(img_user,kp1,img_ref,kp2,matches_fl[:500],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,matchColor=(0,255,0))
        plt.imshow(keypoints_1)
        # plt.show()
        # plt.imshow(keypoints_2)
        # plt.show()
        plt.imshow(img3)
        plt.show()
        print(len(keypoints_1))
        print(all_img_ref,len(matches_fl))
        # print(all_img_ref,len(matches_fl))
    print("Time elapsed: {:.4f}".format(time.time()-start))


if __name__ == "__main__":
    main()