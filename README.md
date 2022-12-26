# Requirement

***
> This code appears to be implementing a hand and object detection system using the OpenCV library and the MediaPipe
> framework.

> The code captures frames from a video stream and processes them to detect and track objects and hands. For object
> detection, it uses a haar cascade classifier to identify faces in the frame. It also uses a function called
> make_mask_for_image to detect objects within a certain color range by creating a mask from the frame and finding
> contours within the mask. For hand detection, the code uses the MediaPipe framework to identify hand landmarks and
> determine the handedness of each detected hand. It also checks for raised fingers on each hand by comparing the
> positions of the hand landmarks to determine if each finger is raised or not.

> The code also includes some visualization features, such as drawing bounding boxes around detected objects and drawing
> hand landmarks and connections on the frame.
***

# Tools

|            Tool             | Version  |
|:---------------------------:|:--------:|
|           Python            |  3.9.13  |
|        OpenCV-Python        | 4.5.4.60 |
|           Imutils           |  0.5.4   |
|          MediaPipe          |  0.8.11  |
|            NumPy            |  1.23.3  |
| google.protobuf.json_format |  4.21.1  |
|        IDE: PyCharm         |  2022.3  |

# Result

## Object detection

![object_detection (Mittel).png](description_pictures%2Fobject_detection%20%28Mittel%29.png)

## Hand detection

![hand_detection_1 (Mittel).png](description_pictures%2Fhand_detection_1%20%28Mittel%29.png)
![hand_detection_2 (Mittel).png](description_pictures%2Fhand_detection_2%20%28Mittel%29.png)

## Face detection

![face_detection (Mittel).png](description_pictures%2Fface_detection%20%28Mittel%29.png)

# Developer

Omar Abdulkhalek

