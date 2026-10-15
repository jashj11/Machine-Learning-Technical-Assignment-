import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(repo_id="dronefreak/bdd100k-yolo11n", filename="best.pt")
model = YOLO(weights)

imgL = cv2.imread("kitti_1.png")
imgR = cv2.imread("kitti_test_2.png")

imgL_g = cv2.imread("kitti_1.png",0)
imgR_g = cv2.imread("kitti_2.png",0)

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
    speckleRange=speckleRange,
)

disp_raw = stereo.compute(imgL_g, imgR_g).astype(np.float32)

disparity_pixels = disp_raw / 16.0

with np.errstate(divide='ignore', invalid='ignore'):
    depth = (focal_length * baseline) / disparity_pixels
    depth[disparity_pixels <= 0] = 0  

np.save("depth_map.npy", depth)
results = model.predict(imgL, conf=0.4, device="cpu")

for box in results[0].boxes:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    
    xyxy = box.xyxy[0].cpu().numpy().astype(int)
    xmin, ymin, xmax, ymax = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    
    depth_roi = depth[ymin:ymax, xmin:xmax]
    valid_depths = depth_roi[(depth_roi > 0) & np.isfinite(depth_roi)]
    
    if valid_depths.size > 0:
        object_distance = np.median(valid_depths)
        print(f"Detected [{cls_name}] at an estimated distance of: {object_distance:.2f} meters")
        
        cv2.rectangle(imgL, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(imgL, f"{cls_name} {object_distance:.2f}m", (xmin, max(ymin - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

disp_vis = cv2.normalize(disp_raw, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

max_visualization_depth = 50.0
depth_clipped = np.clip(depth, 0, max_visualization_depth)

depth_vis = cv2.normalize(depth_clipped, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

cv2.imshow("Disparity Map", disp_vis)
cv2.imshow("Depth Map (Colormapped)", depth_colormap)
cv2.imshow("YOLO + Metric Distance Estimation", imgL)

cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Depth map calculated successfully! Max depth: {np.max(depth):.2f}m")