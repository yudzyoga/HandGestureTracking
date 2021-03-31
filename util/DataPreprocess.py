import numpy as np
import pandas as pd
import random
import os

dataset_fold = "dataset/"

def split_train_test():
    def get_skeleton(path_to_csv):
        """
        This function will return all joints data from skeleton.csv
        Output type: list of array -> len(output) == isincluded frame
        Each array will have (21, 3) of shape
        """
        # read csv file
        df = pd.read_csv(path_to_csv)

        # exclude isincluded == 0
        df = df[df.isincluded==1]

        # iniialize variables
        all_data = []
        joints = []
        last_filename = df.iloc[0].filename

        # get all joints from file
        for index, row in df.iterrows():
            if row.filename != last_filename:
                all_data.append(np.array(joints))
                joints = []
                last_filename = row.filename
            _joint = np.array([row.x, row.y, row.z])
            joints.append(_joint)
        all_data.append(np.array(joints))

        return all_data

    
    
    def parse_file(dataset_path):
        """
        This function will iterate all dataset to get the formatted data
        Output: list of dict
        dict = {'label': integer,
                'skeleton': get_skeleton(path_to_csv)} 
        """
        train_data = []
        test_data = []

        gestures = sorted([folder for folder in os.listdir(dataset_fold) if 'gesture' in folder])
        for gesture in gestures:
            label = gesture.split('_')[-1]
            videos = ['/' + data + '/' for data in os.listdir(dataset_fold + gesture)]
            random.shuffle(videos)
            test_marks = videos[:2]
            for video in videos:
                _dict = {}
                skeleton_path = dataset_fold + gesture + video + 'skeleton.csv'
                if os.path.exists(skeleton_path):
                    _dict['label'] = label
                    _dict['skeleton'] = get_skeleton(skeleton_path)
                    if video in test_marks:
                        test_data.append(_dict)
                    else:
                        train_data.append(_dict)
                else:
                    print("Can't find {}".format(skeleton_path))
                    pass
                
        return train_data, test_data

    
    return parse_file(dataset_fold)   