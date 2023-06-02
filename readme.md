(25th May 2023)
While interning at dorleco, I worked on object detection and multiple object tracking. YOLO-NAS was used for object detection due to its accuracy and greater fps. DeepSORT was used for multiple object tracking. Using the center of bounding box of each detection, depth was calculated using depth camera. And the closest depth was printed.
Why we need tracking?
Without tracking there are high chances that vehicle which are ocluded will not be detected, but if we are tracking them, vehicles which are ocluded will be detected using previous frame.

Sensors used: RGB camera and Depth camera
Software: Carla, ROS
