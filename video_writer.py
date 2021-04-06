import cv2
from stream import *

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    ap.add_argument("--config-file", default="configs/eval_real_world_testset.yaml",
        metavar="FILE", help="path to config file")
    ap.add_argument("opts", help="Modify config options using the command-line",
        default=None, nargs=argparse.REMAINDER)
    ap.add_argument("--device-type", default='realsense', type=str,
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


    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter('test_input3.avi',fourcc, 30.0, (640,480))

    hg = Hand_Graph_CNN(args)

    while True:
        pressedKey = cv2.waitKey(5) & 0xFF
        action = hg.pressed_key(pressedKey)
        
        color, depth = hg.get_frames()

        out.write(color)
        cv2.imshow('frame',color)

        if (action == None):
            pass
        elif (action == 'stop'):
            break
        else:
            print(action)

    hg.end_stream()
    out.release()