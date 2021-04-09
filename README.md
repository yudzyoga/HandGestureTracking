# HandGestureTracking

Commands : 
Inference : python stream.py --inference --save --model <model .pth path> --device-id <video input path>

Inspect : python stream.py --inspect 

Record : python stream.py --record --device-type <'normal' / 'realsense'>

Training :
 - Training from dataset
  ```bash
  python train_on_custom.py -b 64 -lr 1e-3 --dp_rate 0.4 -j 12 --epochs 10000 --patiences 300
  ```
 - Training from pickle
  ```bash
  python train_on_custom.py -b 64 -lr 1e-3 --dp_rate 0.4 -j 12 --epochs 10000 --patiences 300 --load_dict 1
  ```
