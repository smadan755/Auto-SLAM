import cv2
import numpy as np


# ===============================================================================
# --- 1. VISUAL ODOMETRY CLASS 
# ===============================================================================
class VisualOdometry():
    def __init__(self, calib_filepath):
        self.K, self.P = self._load_calib(calib_filepath)
        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, prev_frame, current_frame):
        keypoints1, descriptors1 = self.orb.detectAndCompute(prev_frame, None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(current_frame, None)
        matches = []
        if descriptors1 is not None and descriptors2 is not None:
            matches = self.flann.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            pass
        q1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
        q2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
        return q1, q2, keypoints2

    def get_pose(self, q1, q2):
        if len(q1) < 8:
            return None
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if Essential is None:
            return None
        _, R, t, _ = cv2.recoverPose(Essential, q1, q2, self.K, mask=mask)
        return self._form_transf(R, np.squeeze(t))
