import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list) ):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12)) 
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       min_face_detection_confidence=0.2,
                                       min_face_presence_confidence=0.2,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# STEP 3: Load the input image.
#image = mp.Image.create_from_file("/idiap/temp/pvuillecard/libs/facetorch_extra/datasets/DFEW/examples/11205/frame0001.png")

#print(image)

image = cv2.imread("/idiap/temp/pvuillecard/libs/facetorch_extra/datasets/DFEW/examples/clip_4805/frame0001.png")
rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
# STEP 4: Detect face landmarks from the input image.
detection_result = detector.detect(rgb_frame)

print(detection_result)
# STEP 5: Process the detection result. In this case, visualize it.
#annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)

# save the image 
#cv2.imwrite("/idiap/temp/pvuillecard/libs/facetorch_extra/data/output/testdfew.jpg", annotated_image)

face_landmarks_list = detection_result.face_landmarks[0] # only the first face
print(face_landmarks_list)
ldmrks = []
for idx in range(len(face_landmarks_list) ):
    face_landmarks = face_landmarks_list[idx]
    print(face_landmarks )
    ldmrks += [face_landmarks.x, face_landmarks.y, face_landmarks.z]
  
print(len(ldmrks))
ldmrks = np.array(ldmrks).reshape(-1, 3)
#rescale 
ldmrks[:, 0] = ldmrks[:, 0] * image.shape[1]
ldmrks[:, 1] = ldmrks[:, 1] * image.shape[0]
ldmrks[:, 2] = ldmrks[:, 2] * image.shape[1]

# plot the landmarks in the image 
for ldmrk in ldmrks:
    cv2.circle(image, (int(ldmrk[0]), int(ldmrk[1])), 1, (0, 255, 0), -1)

cv2.circle(image, (int(ldmrks[1, 0]), int(ldmrks[1, 1])), 1, (0, 0, 255), -1)
# save the image 
cv2.imwrite("/idiap/temp/pvuillecard/libs/facetorch_extra/data/output/testdfew.jpg", image)
#print(detection_result.face_blendshapes[0])

blend = []
name = []
for cat in detection_result.face_blendshapes[0]:
    name.append(cat.category_name)
    blend.append(cat.score)
print(len(blend), blend)
print(name)

re = detection_result.facial_transformation_matrixes[0]
re_f = np.array(re).reshape(-1)
print(list(re_f))
print(re)
if re[2, 0] != 1 and re[2, 0] != -1:
    x = np.arcsin(re[2, 0])
    y = np.arctan2(
        re[1, 2] / np.cos(x),
        re[2, 2] / np.cos(x),
    )
    z = np.arctan2(
        re[0, 1] / np.cos(x),
        re[0, 0] / np.cos(x),
    )

else:  # Gimbal lock
    z = 0
    if re[2, 0] == -1:
        x = np.pi / 2
        y = z + np.arctan2(re[0, 1], re[0, 2])
    else:
        x = -np.pi / 2
        y = -z + np.arctan2(-re[0, 1], -re[0, 2])
yaw, pitch, roll = x*180/np.pi, y*180/np.pi, z*180/np.pi

print(yaw, pitch, roll)
