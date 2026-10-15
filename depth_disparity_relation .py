import cv2
import numpy as np

imgL = cv2.imread("kitti_1.png", 0)
imgR = cv2.imread("kitti_2.png", 0)

# Taken directly from cal.txt of KITTI dataset 
P2 = np.array([
    [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
    [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]
])

P3 = np.array([
    [7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, -3.395242000000e+02],
    [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02,  2.199936000000e+00],
    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00,  2.729905000000e-03]
])

focal_length = P2[0, 0]
baseline = abs(P2[0, 3] - P3[0, 3]) / focal_length

minDisparity = 0
numDisparities = 64
blockSize = 9
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 8

stereo = cv2.StereoSGBM_create(
    minDisparity=minDisparity,
    numDisparities=numDisparities,
    blockSize=blockSize,
    disp12MaxDiff=disp12MaxDiff,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange
)

disp_raw = stereo.compute(imgL, imgR).astype(np.float32)

disparity_pixels = disp_raw / 16.0


with np.errstate(divide='ignore', invalid='ignore'):
    depth = (focal_length * baseline) / disparity_pixels
    depth[disparity_pixels <= 0] = 0  


np.save("depth_map.npy", depth)

disp_vis = cv2.normalize(disp_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

max_visualization_depth = 50.0
depth_clipped = np.clip(depth, 0, max_visualization_depth)

depth_vis = cv2.normalize(depth_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

cv2.imshow("Disparity Map", disp_vis)
cv2.imshow("Depth Map (Colormapped)", depth_colormap)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Depth map calculated successfully! Max depth: {np.max(depth):.2f}m")