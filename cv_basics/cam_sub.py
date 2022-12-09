# Import the necessary libraries
import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import os, time, tqdm
import numpy as np
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from visualize import Visualization

class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
      
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      'video_frames', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning
      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
  
  def setup_cfg(path):
    cfg = get_cfg()
    cfg.MODEL.DEVICE='cpu'
    if (VAR_LAYER_CNT == 50):
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    else:
        cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = VAR_NUM_CLASSES  # only has one class (chicken)
    cfg.MODEL.WEIGHTS = os.path.join(path, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    cfg.freeze()
    return cfg

  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data)
    
    # Display image
    #cv2.imshow("camera", current_frame)
    
    setup_logger(name="fvcore")

    r_path = VAR_RES_DIR
    m_path = VAR_OUTPUT_DIR
    cfg = setup_cfg(m_path)
    os.makedirs(r_path, exist_ok=True)

    print('meta: ', cfg.DATASETS.TEST[0], cfg.DATASETS)
    vis = Visualization(cfg)

    idx=0
    while True:
        img = cv2.imread("camera", current_frame)

        num_instances, v_output = vis.run_on_image(img)

        if num_instances == -1:
            continue

        fname = 'img_' + str(idx) + '.png'
        out_filename = os.path.join(r_path, fname)
        idx += 1
        v_output.save(out_filename)

    rs.release()
    cv2.destroyAllWindows()

    cv2.waitKey(1)
  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  image_subscriber = ImageSubscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
