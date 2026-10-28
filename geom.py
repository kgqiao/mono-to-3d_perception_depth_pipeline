# geom.py: depth -> 3D points, the ground plane, and depth metrics (steps D1 to D3).
# Same functions as depthkit.py in the Wing depth-to-3D starter.
import numpy as np
import cv2


def load_depth_png(path):
    """KITTI depth PNG: uint16, meters = value / 256, 0 = no LiDAR return."""
    from PIL import Image
    return np.asarray(Image.open(path), dtype=np.float32) / 256.0

def K_from_P(P):
    """KITTI odometry calib gives 3x4 projection matrices. The left 3x3 block is K."""
    return np.asarray(P, dtype=float)[:3, :3].copy()


def backproject(depth, K):
    """Backprojects 2D Pixels into 3D camera coordinates

    depth: H x W meters (0 = skip). 
    Return (N x 3 points in the camera frame: x right, y down, z forward, u, v)."""
    #1. Find indices of all valid pixels where depth > 0. 
    # v = corresponds to row index (y)
    # u = corresponds to col index (x)
    v, u = np.where(depth > 0) # np.where good for bw pics, auto fills out missing lidar returns (0)

    #2. Extract corresponding depth values (Z) got these N pixels
    z = depth[v, u]

    #3. Create homoegenous coordinates [u, v, 1] for all N pixels
    # Stack along axis 0 to create shape of (3, N)
    hom_pixels = np.stack([u, v, np.ones_like(u)], axis = 0)

    #4. Compute the inverse of 3 x 3 intrinsic matrix
    K_inverse = np.linalg.inv(K)

    #5. Multiply K_inverse by homoegenous pixels to get normalized rays (3, N)
    # @ = matrix multiplication supported by numpy
    rays = K_inverse @ hom_pixels

    #6. Scale each 3D ray vecotr by its respective depth value
    points_3d = rays * z #Ends with a scaled accurate collection of 3d ray vectors

    #7. Transpose from (3, N) scale to desired (N, 3) matrix shape
    points_3d_transposed = points_3d.T

    # (D1): the pinhole model run backwards: X = (u - cx) Z / fx, Y = (v - cy) Z / fy
    return points_3d_transposed


def ransac_plane(pts, thresh = 0.10, iters = 500, seed = 0):
    """
    Sample 3 points, plane through them, count points within thresh, 
    keep the best, refit with SVD, flip so d > 0

    Find the ground plane from your 3d points (of the ojbect)
    AKA define a flat plane in 3d space
    1. Sample 3 points
    2. Finding the tilt (normal vector)
    2. Measure distance to every single point
    3. Count the votes (compare distance to specific threshold of distnace)
        - If point is within 10cm of the plane, it "votes" for it = marked
         as an inlier (part of the plan)
        - If point is further than 10cm away, (ex: car, tree, wall), ignored
          as an outlier
    4. Remember the plane that got the most inliers out of all planes (across iters = 500)
    5. Flip arrow of that chosen plane's vector upwards
        - In the KITTI dataset coordinate system, the camera's vertical axis 
            (/(Y/)) points downward toward the ground. 
        - Therefore, a proper ground plane should have an arrow pointing up 
            toward the sky, meaning its /(b/) value must be negative.
        - If our winning plane happens to have its arrow pointing down into the 
            earth, we multiply the whole equation by /(-1/) to flip it right-side 
            up without changing the actual flat surface.    

    Fit a*x + b*y + c*z + d = 0 with |(a, b, c)| = 1 and the normal 
    pointing toward the camera (d > 0).
    Return (np.array([a, b, c, d]), inlier mask)."""

    # Set failure condition as to not break RANSAC calculations later
    # Test Failure Condition A: When there's not enough points to run RANSAC
    if len(pts) < 3:
        return None, None

    #1. Set the random seed for reproducibility
    set_range = np.random.default_rng(seed)

    num_pts = pts.shape[0]
    best_inliers = -1
    best_model = None
    best_mask = None

    for i in range(iters):
        #1. Randomly sample 3 unique points
        idx = set_range.choice(num_pts, size = 3, replace = False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]

        # 2. Compute vectors spanning the plane
        v1 = p2 - p1
        v2 = p3 - p1

        # 3. Compute the normal vector using the cross product
        normal = np.cross(v1, v2) # Finds a perpendicular normal vector to the plane formed by the 3 sampled points.
        normal_val = np.linalg.norm(normal) # Calculate the Euclidean length (magnitude) of the vector normal
        if normal_val < 1e-6: # Skip collinear points (where cross product magnitude = 0)
            continue

        # Conditional SKIP: Skip if bad sample setups --
        if normal_val < 1e-6:
            continue

        # 3a. Normalize the normal vector to ensure |(a, b, c)| = 1
        normal = normal / normal_val
        a, b, c = normal #split the points into three form normalized vector now
        # Normalize = Scaling a vector to the length of 1

        # 3b. Calculate d parameter (d = -n *dot product p)
        d = -np.dot(normal, p1)

        # 4. Compute absolute distance from all points to the chosen plane
        # Distance formula: |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
        # Since |(a, b, c)| = 1, the denominator is just 1
        distances = np.abs(pts @ normal + d) #Uses a fast dot product across all \(N\) points simultaneously to measure how far away every point lies from the candidate plane.

        #5. Identify inliers within the desired threshold
        mask = distances < thresh
        num_inliers = np.sum(mask)

        #6. Save the model if it has the most inliers found so far
        if num_inliers > best_inliers: #if curr model's infliers have more, re-save the best
            best_inliers = num_inliers
            best_model = np.array([a, b, c, d])
            best_mask = mask

        # 7. Orient normal vector properly
        # KITTI: Y points down, so a ground plane normal pointing up must have b < 0
        
        
        # Set failure condition as to not break RANSAC calculations later
        # Test Failure Condition B: When there's not enough points to run RANSAC
        if best_model is None: 
            return None, None

        # Orient the normal vector properly
        if best_model[1] > 0:
            best_model = -best_model # Adding negative to reverse

    return best_model, best_mask


def height_above(pts, plane):
    """
    Calculates the signed distance from the 3D points to th plane.
    (+ = on the camera's side, i.e. above the road)
    
    pts: N x 3 array of 3D points
    plane: np.array([a, b, c, d]) where |(a, b, c)| = 1
    
    Returns: A N-dimensional array of signed distances (heights)"""
    
    # 1. Unpack the plane coefficients
    a, b, c, d = plane
    normal = np.array([a, b, c]) # [a, b, c] is the nomral vector, d = distance of how floor is shifted away from camera

    # 2. Compute the signed distance for all points simultaneously
    # # Distance = (a*x + b*y + c*z + d)
    heights = pts @ normal + d #a dot product with the normal, plus d

    return heights 
    

def metrics(pred, gt, max_depth=80.0):
    """
    Evaluate how accurately the predicted depth map matches ground-truth/actual depth data using classic metrics (in the dataset's benchmarks)

    1. Create filter mask and count pixels - so that valid pixels are evaluated
    (Since liDAR and depth sensors have physical limits)
        - Build a boolean mask checking 3 required conditions:
            1. Ground truth > 0
            2. Ground truth < max_depth (80 in this case)
            3. Prediction > 0 
        - Sum up all values to find exact number of valid pixels evaluated
    2. Extract valid values - squeeze images into flat 1D arrays containing only those valid pixels
    3. Find Absolute Relative Error (AbsRel)
    - AbsRel = Measures the size of the depth error relative to - object's true distance
    4. Find Root Mean Squared Error (RMSE) = high penalization for large mistakes
    5. Find Delta 1 Threshold Accuracy
    Delta 1 = measures % of your piexels that are highly acccurate.
    Checks if ratio between prediction vs ground truth is within 25% margin
    
    Only where 0 < gt < max_depth and pred > 0. 
    Return dict abs_rel, rmse, delta1, n_pixels."""

    # 1. Create bool filter mask to isolate the valid evaluation pixels
    mask = (gt > 0) & (gt < max_depth) & (pred > 0) #initialize mask

    # 1a. Count number of pixels that match criteria of mask
    num_pixels = int(np.sum(mask))
    pred = int(np.sum(mask)) # Apply same mask to pred
    
    if num_pixels == 0: # Handle edge case where no pixesl pass thru depth mask
        return {
            "abs_rel": 0.0,
            "rmse": 0.0,
            "delta1": 0.0,
            "n_pixels": 0
        }

    # 3. Computer Absolute Relative Error (AbsRel)
    abs_rel = np.mean(np.abs(pred - gt) / gt)

    # 4. Compute Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean(pred - gt) ** 2)

    # 5. Compute the Delta 1 threshold accuracy
    ratio = np.maximum(pred / gt, gt / pred)
    delta1 = np.mean(ratio < 1.25)

    # 6. Package and return the metrics dictionary
    return {
        "abs_rel": float(abs_rel),
        "rmse": float(rmse),
        "delta1": float(delta1),
        "n_pixels": num_pixels
    }


def garg_crop(h, w):
    """The standard Eigen-split evaluation crop (Garg et al.). Returns a boolean H x W mask."""
    m = np.zeros((h, w), bool)
    m[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True

    return m
