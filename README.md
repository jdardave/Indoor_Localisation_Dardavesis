# Indoor localisation and location tracking in semi-public buildings based on LiDAR point clouds and images of the ceilings
## Ioannis Dardavesis

### File explanation
- data: Includes point cloud and image datasets that were acquired for this thesis. The point clouds where acquired with a LiDAR sensor of an iPad 12 pro, while the cameras with a Xiaomi Redmi Note 9s. You can find data that were used as reference and different user data in the corresponding files.
  - database: Contains the point clouds that were acquired with the Sitescape app and the LiDAR sensor of the iPad and were used as reference.
  - user: Contains the user point clouds that were acquired with the Sitescape app and the LiDAR sensor of the iPad.
  - pix4d_database: Contains the point clouds that were acquired with the Pix4d Catch app and the LiDAR sensor of the iPad and were used as reference.
  - pix4d_user: Contains the user point clouds that were acquired with the Pix4d Catch app and the LiDAR sensor of the iPad.
  - single_image_database: Contains images from different rooms that were used as reference to implement different image matching techniques in image_matching.py
  - single_image_user: Contains user images from different rooms that were used to implement different image matching techniques in image_matching.py
  - pc_centers: Includes the coordinates of the centers of different users point clouds after different registrations. This data is used to create the scatter plots to compare the registration between the centers of the reference point cloud for a room and the centers of different tested point clouds. This data is used in scatter.py
- code: Includes the coding and algorithms that were used for this thesis
  - App: Web-application for indoor localisation that was developed using Flask. CSS and HTML files that were used for styling the web-app are also included in the templates folder, while the main code in the app.py
  - image_matching.py: Includes different combinations of feature detectors, descriptors and matching techniques from image data. The data that are needed for the results to be reproduced, can be found in the single_image_database and single_image_user folders.
  - point_cloud_registration.py: Includes different combinations of global and local registration techniques from point cloud data. The data in the folders pix4d_database and pix4d_user can be used to reproduce results from point clouds that were acquired with Pix4d Catch, while the ones in database and user folders to reproduce results from point clouds that were acquired with SiteScape app.
  - scatter.py: Code that creates scatter plots from the coordinates of the centers of a reference and different user point clouds after registration for a specific room






