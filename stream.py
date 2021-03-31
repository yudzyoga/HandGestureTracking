# Run this code by
# python stream.py --render

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import time
import glob
import os
import math
import torch
import argparse
import numpy as np
import os.path as osp
import pyrealsense2 as rs
import mediapipe as mp
import csv
from typing import Union, Tuple, List

class Hand_Graph_CNN():
    def __init__(self, args, cam_img_width=640, cam_img_height=480):
        self.args = args
        
        # initilaize system
        self.render_output = not args.no_render
        self.image_count = -1
        self.dataset_path = './dataset'
        self.num_gesture = 6
        self.data_dir_name = self.setup_dirs(self.dataset_path, self.num_gesture)
        self.img_width = cam_img_width
        self.img_height = cam_img_height
        self.xyz_pos_list = []
        self.uv_pos_list = []

        # record system on
        self.isRecordMode = args.record
        self.isRecording = False
        self.isRecording_temp = False
        self.recordStatus = 1

        # inspect mode on
        self.isInspectMode = args.inspect
        self.inspectStatus = 0
        self.resetInspect = True
        self.imgStatus = 0
        self.resetInspectImage = True

        # dataset stored
        self.dataset_ids = None
        self.dataset_path = os.path.join('.', 'dataset')
        self.gesture_id = None
        self.data_id = None

        self.on_loop = False
        self.filenames = []
        self.csv_x = []
        self.csv_y = []
        self.csv_z = [] 
        self.csv_u = []
        self.csv_v = []
        self.csv_isincluded = [] 

        if (self.args.device_type == 'normal'):
            self.cap = cv2.VideoCapture(self.args.device_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_img_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_img_height)
            
        elif (self.args.device_type == 'realsense'):
            self.pipeline = rs.pipeline()

            # Create a config and configure the pipeline to stream
            #  different resolutions of color and depth streams
            config = rs.config()

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            device = pipeline_profile.get_device()
            device_product_line = str(device.get_info(rs.camera_info.product_line))

            # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.depth, cam_img_width, cam_img_height, rs.format.z16, 30)
            if device_product_line == 'L500':
                config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            else:
                config.enable_stream(rs.stream.color, cam_img_width, cam_img_height, rs.format.bgr8, 30)

            # Start streaming
            profile = self.pipeline.start(config)

            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()

            # We will be removing the background of objects more than
            #  clipping_distance_in_meters meters away
            clipping_distance_in_meters = 1 #1 meter
            clipping_distance = clipping_distance_in_meters / depth_scale

            # Create an align object
            # rs.align allows us to perform alignment of depth frames to others frames
            # The "align_to" is the stream type to which we plan to align depth frames.
            align_to = rs.stream.color
            self.align = rs.align(align_to)

    def setup_dirs(self, path, num_gesture):
        def diff(li1, li2):
            return (list(list(set(li1)-set(li2)) + list(set(li2)-set(li1))))
        def create_dirs(path, gesture_dirs_not_exist):
            for (i, i_dir) in enumerate(gesture_dirs_not_exist):
                os.mkdir(os.path.join(path, 'gesture_{}'.format(i_dir)))
        def count_data_dirs(path, gesture_i):
            data_num = [int(fullpath[-1]) for fullpath in glob.glob(os.path.join(path, f'gesture_{gesture_i}', 'data*'))]
            if(len(data_num) > 0):
                return max(data_num)
            else:
                return 0

        if not os.path.exists(path):
            print("creating dir {}".format(path))
            os.mkdir(path)

        # check the existing dirs 
        gesture_dirs_exist = glob.glob(os.path.join(path, 'gesture_*'))
        gesture_dirs_exist = [int(dirs[-1]) for dirs in gesture_dirs_exist]
        target_gesture_dirs = [int(i_dir) for i_dir in range(1, num_gesture + 1)]
        gesture_dirs_not_exist = diff(target_gesture_dirs, gesture_dirs_exist)

        if(len(gesture_dirs_not_exist) > 0):
            create_dirs(path, gesture_dirs_not_exist)
        
        # count the data folder inside each gestures
        gesture_data_count = [count_data_dirs(path, i) + 1 for i in target_gesture_dirs]
        
        return gesture_data_count

    def get_frames(self):
        if (self.args.device_type == 'normal'):
            ret, frame = self.cap.read()
            if ret:
                return frame, None
            
        elif (self.args.device_type == 'realsense'):
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames() # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)

            return color_image, depth_image 

    def detect(self, hands, image, depth=None):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        stat = False
        xyz_pos, uv_pos = None, None
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # check detection
        if results.multi_hand_landmarks:
            stat = True
            xyz_pos, uv_pos = self.extract_coordinates(results)
        
        return image, results, stat, xyz_pos, uv_pos
    
    def extract_coordinates(self, results):
        def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, \
                                            image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
            """Converts normalized value pair to pixel coordinates."""
            # Checks if the float value is between 0 and 1.
            def is_valid_normalized_value(value: float) -> bool:
                return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

            if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
                # TODO: Draw coordinates even if it's outside of the image bounds.
                return None
            
            x_px = min(math.floor(normalized_x * image_width), image_width - 1)
            y_px = min(math.floor(normalized_y * image_height), image_height - 1)
            return [x_px, y_px]

        xyz_pos = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
        uv_pos = [_normalized_to_pixel_coordinates(p[0], p[1], self.img_width, self.img_height) for p in xyz_pos]

        return xyz_pos, uv_pos
        

    def show_result(self, mp_drawing, mp_hands, image, results):
    # show results
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if(self.isRecordMode):
                if(self.isRecording):
                    cv2.putText(image, f'Recording : Gesture{self.recordStatus} - Take{self.data_dir_name[self.recordStatus - 1]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    cv2.putText(image, f'Not recording : Gesture{self.recordStatus} - Take{self.data_dir_name[self.recordStatus - 1]}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('MediaPipe Hands', image)            

    def store_data(self, image, xyz_pos, uv_pos):
        self.image_count += 1
        self.xyz_pos_list.append(xyz_pos)
        self.uv_pos_list.append(uv_pos)
        path = os.path.join('.', 'dataset', 'gesture_{}'.format(self.recordStatus), 'data{}'.format(self.data_dir_name[self.recordStatus - 1]))
        if not os.path.exists(path):
            print("creating dir {}".format(path))
            os.mkdir(path)
        cv2.imwrite(os.path.join(path, '{0:07}.jpg'.format(self.image_count)), image)

    def write_data(self):
        path = os.path.join('.', 'dataset', 'gesture_{}'.format(self.recordStatus), 'data{}'.format(self.data_dir_name[self.recordStatus - 1]))
        filepath = os.path.join(path, 'skeleton.csv')
        with open(filepath, "w+") as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'joint', 'x', 'y', 'z', 'u', 'v', 'isincluded'])
            for (i_data, xyz_list) in enumerate(self.xyz_pos_list):
                for (i_lm, xyz) in enumerate(xyz_list):
                    x, y, z = self.xyz_pos_list[i_data][i_lm]
                    u, v = self.uv_pos_list[i_data][i_lm]
                    writer.writerow(['{0:07}.jpg'.format(i_data), i_lm, x, y, z, u, v, 1])
        self.image_count = -1
        self.xyz_pos_list = []
        self.uv_pos_list = []
        self.data_dir_name[self.recordStatus - 1] += 1

    def pressed_key(self, key):
        if(self.isRecordMode):
            if(key == ord('1') and self.recordStatus != 1):
                self.recordStatus = 1
                self.isRecording = False
                return 'activate_mode_1'
            elif(key == ord('2') and self.recordStatus != 2):
                self.recordStatus = 2
                self.isRecording = False
                return 'activate_mode_2'
            elif(key == ord('3') and self.recordStatus != 3):
                self.recordStatus = 3
                self.isRecording = False
                return 'activate_mode_3'
            elif(key == ord('4') and self.recordStatus != 4):
                self.recordStatus = 4
                self.isRecording = False
                return 'activate_mode_4'
            elif(key == ord('5') and self.recordStatus != 5):
                self.recordStatus = 5
                self.isRecording = False
                return 'activate_mode_5'
            elif(key == ord('6') and self.recordStatus != 6):
                self.recordStatus = 6
                self.isRecording = False
                return 'activate_mode_6'
            elif(key == ord('q')):
                self.isRecording = False
                return 'stop'
            elif(key == 32):
                self.isRecording = not self.isRecording
                return 'record {}'.format(self.isRecording)
        elif(self.isInspectMode):
            if(key == ord('a')):
                self.resetInspectImage = True
                self.imgStatus -= 1
                return 'prev_image'
            elif(key == ord('d')):
                self.resetInspectImage = True
                self.imgStatus += 1
                return 'next_image'
            elif(key == ord('w')):
                self.inspectStatus += 1
                self.resetInspectImage = True
                self.resetInspect = True
                self.imgStatus = 1
                return 'prev_data'
            elif(key == ord('s')):
                self.inspectStatus -= 1
                self.resetInspectImage = True
                self.resetInspect = True
                self.imgStatus = 1
                return 'next_data'
            elif(key == ord('q')):
                self.resetInspectImage = True
                self.resetInspect = True
                return 'stop'
            elif(key == 32):
                self.resetInspectImage = True
                self.csv_isincluded[self.imgStatus] = int(not int(self.csv_isincluded[self.imgStatus]))
                return None
        else:
            if(key == ord('q')):
                return 'stop'
            return None

    def reload_dataset(self):
        # save current dataset
        if(self.on_loop):
            self.write_data()
        self.on_loop = True
        self.resetInspect = False
        self.inspectStatus = self.inspectStatus % len(self.dataset_ids)
        self.gesture_id, self.data_id = self.dataset_ids[self.inspectStatus]
        csv_dir = os.path.join(self.dataset_path, f'gesture_{self.gesture_id}', f'data{self.data_id}')
        self.filenames, self.csv_x, self.csv_y, self.csv_z, \
            self.csv_u, self.csv_v, self.csv_isincluded = load_csv(csv_dir)

    def reload_image(self):
        self.resetInspectImage = False
        self.imgStatus = self.imgStatus % len(self.filenames)
        print('gesture {}, data {}, image {}'.format(self.gesture_id, self.data_id, self.imgStatus))
        img_path = os.path.join(self.dataset_path, f'gesture_{self.gesture_id}', f'data{self.data_id}', f'{self.filenames[self.imgStatus]}')
        img = cv2.imread(img_path)
        if(int(self.csv_isincluded[self.imgStatus])):
            cv2.putText(img, f'Gesture {self.gesture_id} - Data {self.data_id} - Image {self.imgStatus} : Included', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(img, f'Gesture {self.gesture_id} - Data {self.data_id} - Image {self.imgStatus} : Not included', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        return img

    def print_record_status(self):
        print(self.recordStatus)
            
    def end_stream(self):
        if(self.args.device_type == 'normal'):
            self.cap.release()
            cv2.destroyAllWindows()
        
        elif(self.args.device_type == 'realsense'):        
            self.pipeline.stop()

    def write_data(self):
        path = os.path.join('.', 'dataset', f'gesture_{self.gesture_id}', f'data{self.data_id}')
        filepath = os.path.join(path, 'skeleton.csv')
        with open(filepath, "w+") as file:
            writer = csv.writer(file)
            writer.writerow(['filename', 'joint', 'x', 'y', 'z', 'u', 'v', 'isincluded'])
            for i_img in range(len(self.filenames)):
                for i_pt in range(len(self.csv_x[0])):
                    filename, x, y, z, u, v, included = \
                        self.filenames[i_img], self.csv_x[i_img][i_pt], self.csv_y[i_img][i_pt], \
                        self.csv_z[i_img][i_pt], self.csv_u[i_img][i_pt], self.csv_v[i_img][i_pt], \
                        self.csv_isincluded[i_img]
                    writer.writerow([filename, i_pt, x, y, z, u, v, included])

        # path = os.path.join('.', 'dataset', 'gesture_{}'.format(self.recordStatus), 'data{}'.format(self.data_dir_name[self.recordStatus - 1]))
        # filepath = os.path.join(path, 'skeleton.csv')
        # with open(filepath, "w+") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['filename', 'joint', 'x', 'y', 'z', 'u', 'v', 'isincluded'])
        #     for (i_data, xyz_list) in enumerate(self.xyz_pos_list):
        #         for (i_lm, xyz) in enumerate(xyz_list):
        #             x, y, z = self.xyz_pos_list[i_data][i_lm]
        #             u, v = self.uv_pos_list[i_data][i_lm]
        #             writer.writerow(['{0:07}.jpg'.format(i_data), i_lm, x, y, z, u, v, 1])

def load_csv(path):
    csv_filename = []
    csv_x, csv_y, csv_z, csv_u, csv_v, csv_isincluded = [], [], [], [], [], []
    csv_x_arr, csv_y_arr, csv_z_arr, csv_u_arr, csv_v_arr, csv_isincluded_arr = [], [], [], [], [], []

    with open(os.path.join(path, 'skeleton.csv')) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for i, row in enumerate(csv_reader):
            if(i > 0):
                csv_x_arr.append(row[2])
                csv_y_arr.append(row[3])
                csv_z_arr.append(row[4])
                csv_u_arr.append(row[5])
                csv_v_arr.append(row[6])
                # csv_isincluded_arr.append(row[7])
                if((i) % 21 == 0):
                    csv_filename.append(row[0])
                    csv_isincluded.append(row[7])#csv_isincluded_arr)
                    csv_x.append(csv_x_arr)
                    csv_y.append(csv_y_arr)
                    csv_z.append(csv_z_arr)
                    csv_u.append(csv_u_arr)
                    csv_v.append(csv_v_arr)
                    csv_x_arr, csv_y_arr, csv_z_arr, csv_u_arr, \
                        csv_v_arr, csv_isincluded_arr = [], [], [], [], [], []
    return csv_filename, csv_x, csv_y, csv_z, csv_u, csv_v, csv_isincluded

def main_inspect(args):
    hg = Hand_Graph_CNN(args)

    # find each generated gesture and 
    id_list = [[int(string.split('/')[2][8:]), int(string.split('/')[3][4:])] for string in glob.glob('./dataset/gesture_*/data*')]
    id_arr = np.array(id_list)

    if(len(id_arr) == 0):
        pass

    # load csv data
    id_arr = id_arr[np.lexsort(np.fliplr(id_arr).T)]
    dataset_path = os.path.join('.', 'dataset')
    
    hg.dataset_ids = id_arr
    hg.dataset_path = dataset_path
    
    while True:
        pressedKey = cv2.waitKey(5) & 0xFF
        action = hg.pressed_key(pressedKey)
        
        if(hg.resetInspect):
            hg.reload_dataset()

        if(hg.resetInspectImage):
            img = hg.reload_image()

        cv2.imshow('Inspector View', img)
        if (action == None):
            pass
        elif (action == 'stop'):
            break
        else:
            print(action)

    hg.end_stream()
      

def main(args):
    hg = Hand_Graph_CNN(args)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    all_data = []
    action = None
    
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while True:
            pressedKey = cv2.waitKey(5) & 0xFF
            action = hg.pressed_key(pressedKey)

            # calculate time
            start_time = time.time()

            # get frames
            image, depth = hg.get_frames()
            
            # process image
            image, results, stat, xyz_pos, uv_pos = hg.detect(hands, image, None)

            # record
            if(stat and hg.isRecordMode and hg.isRecording):
                hg.store_data(image, xyz_pos, uv_pos)
            elif(stat and hg.isRecordMode and not hg.isRecording and hg.isRecording_temp):
                hg.write_data()

            # show results
            if(hg.render_output):
                hg.show_result(mp_drawing, mp_hands, image, results)

            end_time = time.time()
            FPS = (1 / (end_time - start_time))
            print ("FPS : {:.2f}".format(FPS))

            if (action == None):
                pass
            elif (action == 'stop'):
                break
            else:
                print(action)

            hg.isRecording_temp = hg.isRecording

        hg.end_stream()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    ap.add_argument("--config-file", default="configs/eval_real_world_testset.yaml",
        metavar="FILE", help="path to config file")
    ap.add_argument("opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER)
    ap.add_argument("--device-type", default='normal', type=str,
    	help="input device")
    ap.add_argument("--device-id", default=0, type=int,
        help="device id")
    ap.add_argument("--no-render", action='store_true',
        help="view the rendered output")
    ap.add_argument("--record", action='store_true',
        help="record the training set")
    ap.add_argument("--inspect", action='store_true',
        help="inspect the training set")
    args = ap.parse_args()

    if(args.inspect):
        main_inspect(args)
    else:
        main(args)