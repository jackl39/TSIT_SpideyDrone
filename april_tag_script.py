import cv2
import apriltag
import numpy as np

# Camera matrix and distortion coefficients
CAMERA_MATRIX = np.array([
    [994.59857766, 0.0, 456.70223529],
    [0.0, 997.88115086, 330.56694309],
    [0.0, 0.0, 1.0]
])
DIST_COEFFS = np.array([0.00192996, 0.58087075, -0.0120431, -0.00259187, -1.78403902])
TAG_SIZE = 0.052

def main():
    # Change this to a rostopic 
    cap = cv2.VideoCapture(4)

    # Set the resolution to 720p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create AprilTag detector object
    detector = apriltag.Detector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        undistorted_frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        results = detector.detect(gray)

        # Draw results and calculate the pose
        for result in results:
            (ptA, ptB, ptC, ptD) = result.corners
            ptA = (int(ptA[0]), int(ptA[1]))
            ptB = (int(ptB[0]), int(ptB[1]))
            ptC = (int(ptC[0]), int(ptC[1]))
            ptD = (int(ptD[0]), int(ptD[1]))

            cv2.line(undistorted_frame, ptA, ptB, (0, 255, 0), 2)
            cv2.line(undistorted_frame, ptB, ptC, (0, 255, 0), 2)
            cv2.line(undistorted_frame, ptC, ptD, (0, 255, 0), 2)
            cv2.line(undistorted_frame, ptD, ptA, (0, 255, 0), 2)

            tag_id = result.tag_id
            cv2.putText(undistorted_frame, f"ID: {tag_id}", (ptA[0], ptA[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            fx, fy, cx, cy = CAMERA_MATRIX[0, 0], CAMERA_MATRIX[1, 1], CAMERA_MATRIX[0, 2], CAMERA_MATRIX[1, 2]
            pose, e0, e1 = detector.detection_pose(result, (fx, fy, cx, cy), TAG_SIZE)

            # Check if pose estimation is valid before proceeding
            if pose is not None and pose.shape == (4, 4):  # Ensuring the pose matrix is 4x4
                R = pose[:3, :3]  # Extract rotation matrix
                t = pose[:3, 3]   # Extract translation vector
                distance = np.linalg.norm(t)
                cv2.putText(undistorted_frame, f"Distance: {distance:.2f}m", (ptA[0], ptA[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Frame", undistorted_frame)

        # Break the loop with the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
