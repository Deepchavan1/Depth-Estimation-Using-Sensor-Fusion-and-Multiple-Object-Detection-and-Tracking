import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time


from tracking_helpers import read_class_names, create_box_encoder
from deep_sort import preprocessing, nn_matching
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection



class yolonas_deepSORT:
    '''
    Detector:- YOLO nas
    tracker:- deepSORT
    '''
    def __init__(self, deepSORT_path, detector, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0, coco_names_path:str ="./io_data/input/classes/coco.names"):
        '''
        args: 
            deepSORT_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.detector = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()
        self.detection_boxes = []
        self.detected_class_names = []

        # initialize deep sort
        self.encoder = create_box_encoder(deepSORT_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker
        # self.custom_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        #  'fire hydrant', 'stop sign', 'parking meter', 'bench']

    
    def track(self, frame, output=None, frame_number=None, count_objects=True, save_outputs=False):
        
        start_time = time.time()

        if save_outputs:
            if not os.path.exists(output):
                os.makedirs(output)
        

        # ----------------------------------------- YOLO nas detector -----------------------------------------------------------------


        out = self.detector.predict(frame, conf=0.6)
        preds = out._images_prediction_lst

        for pred in preds:
            yolo_dets = pred.prediction

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0  
        else:
            bboxes = yolo_dets.bboxes_xyxy
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            scores = yolo_dets.confidence
            classes = yolo_dets.labels

            # rem_list = []
        
            # for i in range(len(classes)):
            #     if not classes[i] in self.custom_classes:
            #         rem_list.append(i)
            
            # for i in rem_list:
            #     # print(classes)
            #     classes = np.concatenate((classes[:i],classes[i+1:]))
            #     # print(classes)
            #     scores = np.concatenate((scores[:i],scores[i+1:]))
            #     # print(bboxes)
            #     bboxes = np.concatenate((bboxes[:i , :],bboxes[i+1: , :]))
            #     # print(bboxes)

            num_objects = bboxes.shape[0]




        # ----------------------------------------- Detection completed -----------------------------------------------------------------

        names = []
        for i in range(num_objects): # loop through objects and use class index to get class name
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            names.append(class_name)

        names = np.array(names)
        count = len(names)

        # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------

        features = self.encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

        cmap = plt.get_cmap('tab20b') #initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices] 

        # print(detections)

        self.tracker.predict()  # Call the tracker
        self.tracker.update(detections) #  updtate using Kalman Gain


        for track in self.tracker.tracks:  # update new findings AKA tracks
            # print("ENTERED")
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     print("EXITED")
            #     continue 
            bbox = track.to_tlbr()
            self.detection_boxes.append(bbox)
            class_name = track.get_class()
            self.detected_class_names.append(class_name)
    
            color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
            color = [i * 255 for i in color]
            print("-------------------------------------------------")
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
            cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
            cv2.circle(frame, (int((bbox[0]+bbox[2])//2),int((bbox[1]+bbox[3])//2)), radius=0, color=(0, 0, 255), thickness=20)
            print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

        # for detection in detections:
        #     bbox = detection.to_tlbr()
        #     class_name = detection.get_class()
        #     # print(list(self.class_names.values()))
        #     track_id = list(self.class_names.values()).index(class_name)

        #     color = colors[int(track_id) % len(colors)]  # draw bbox on screen
        #     color = [i * 255 for i in color]
        #     print("-------------------------------------------------")
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        #     cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track_id)))*17, int(bbox[1])), color, -1)
        #     cv2.putText(frame, class_name + " : " + str(track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    

        #     print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
       

        # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
    
        fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
        # if save_outputs: print(f"Processed frame no: {frame_number} || Current FPS: {round(fps,2)}")
        # else: print(f"Current FPS: {round(fps,2)} || Objects tracked: {count}")
        
        if count_objects:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
            cv2.putText(frame, "FPS: {}".format(round(fps,2)), (5, 65), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)


        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if save_outputs:
            cv2.imwrite(output + "/{}.jpg".format(frame_number), result)

        cv2.destroyAllWindows()

        return_bboxs = self.detection_boxes
        return_class_names = self.detected_class_names

        self.detection_boxes = []
        self.detected_class_names = []

        return result, return_bboxs, return_class_names
    