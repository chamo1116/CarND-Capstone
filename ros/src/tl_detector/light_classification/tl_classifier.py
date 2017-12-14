from styx_msgs.msg import TrafficLight
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
from glob import glob
import os
from keras.models import load_model

class TLClassifier(object):
    def __init__(self):

        cwd = os.path.dirname(os.path.realpath(__file__))

        # load keras Lenet style model from file
        self.class_model = load_model(cwd+'/model.h5')
        self.class_graph = tf.get_default_graph()

        # detection graph
        self.dg = tf.Graph()
        # load 
        with self.dg.as_default():
            gdef = tf.GraphDef()
            with open(cwd+"/models/frozen_inference_graph.pb", 'rb') as f:
                gdef.ParseFromString( f.read() )
                tf.import_graph_def( gdef, name="" )

            #get names of nodes. from https://www.activestate.com/blog/2017/08/using-pre-trained-models-tensorflow-go
            self.session = tf.Session(graph=self.dg )
            self.image_tensor = self.dg.get_tensor_by_name('image_tensor:0')
            self.detection_boxes =  self.dg.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.dg.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.dg.get_tensor_by_name('detection_classes:0')
            self.num_detections    = self.dg.get_tensor_by_name('num_detections:0')

        self.tlclasses = [ TrafficLight.RED, TrafficLight.YELLOW, TrafficLight.GREEN ]
        self.tlclasses_d = { TrafficLight.RED : "RED", TrafficLight.YELLOW:"YELLOW", TrafficLight.GREEN:"GREEN", TrafficLight.UNKNOWN:"UNKNOWN" }

        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light. OpenCV is BGR by default.

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        box = self.localize_lights( image )
        if box is None:
            return TrafficLight.UNKNOWN
        class_image = cv2.resize( image[box[0]:box[2], box[1]:box[3]], (32,32) )
        return self.classify_lights( class_image )



    def classify_lights(self, image):
        """ Given a 32x32x3 image classifies it as red, greed or yellow
            Expects images in BGR format. Important otherwide won't classify correctly
            
        """
        status = TrafficLight.UNKNOWN
        img_resize = np.expand_dims(image, axis=0).astype('float32')
        with self.class_graph.as_default():
            predict = self.class_model.predict(img_resize)
            status  = self.tlclasses[ np.argmax(predict) ]

        return status



    def localize_lights(self, image):
        """ Localizes bounding boxes for lights using pretrained TF model
            expects BGR8 image
        """

        with self.dg.as_default():
            #switch from BGR to RGB. Important otherwise detection won't work
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            tf_image_input = np.expand_dims(image,axis=0)
            #run detection model
            (detection_boxes, detection_scores, detection_classes, num_detections) = self.session.run(
                    [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                    feed_dict={self.image_tensor: tf_image_input})

            detection_boxes = np.squeeze(detection_boxes)
            detection_classes = np.squeeze(detection_classes)
            detection_scores = np.squeeze(detection_scores)


            #print(detection_classes)
            #print(detection_scores)
            #print(detection_boxes)

            ret = None
            detection_threshold = 0.3

            # Find first detection of signal. It's labeled with number 10
            idx = -1
            for i, cl in enumerate(detection_classes.tolist()):
                if cl == 10:
                    idx = i;
                    break;

            if idx == -1:
                pass  # no signals detected
            elif detection_scores[idx] < detection_threshold:
                pass # we are not confident of detection
            else:
                dim = image.shape[0:2]
                box = self.from_normalized_dims__to_pixel(detection_boxes[idx], dim)
                box_h, box_w  = (box[2] - box[0], box[3]-box[1] )
                if (box_h <20) or (box_w<20):  
                    pass    # box too small 
                elif ( box_h/box_w <1.6):
                    pass    # wrong ratio
                else:
                    print('detected bounding box: {} conf: {}'.format(box, detection_scores[idx]))
                    ret = box

        return ret
        
    def from_normalized_dims__to_pixel(self, box, dim):
            height, width = dim[0], dim[1]
            box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
            return np.array(box_pixel)


    def draw_box(self, img, box):
        cv2.rectangle(img, (box[1],box[0]), (box[3],box[2]), (255,0,0), 5)
        return img



if __name__ == '__main__':
    if False:
        # test localizatoin
        cl = TLClassifier()
        #img = np.asarray( Image.open('images/3.jpg'), dtype="uint8" )
        img = cv2.imread('images/1.jpg')
        box = cl.localize_lights( img )
        if(box is None):
            print("Couldn't locate lights")
        else:
            cv2.imwrite("images/out.jpg", cl.draw_box( img, box))


    if False:
        #preprocess training images. produce 32x32 images that don't contain background
        for i in range(3):
            paths = glob(os.path.join('classifier_images/labeled_original/{}'.format(i), '*.png'))
            for path in paths:
                print(path)
                img = cv2.imread(path)
                crop = img[3:29, 11:22]
                dst = cv2.resize( crop, (32, 32) )
                cv2.imwrite("prep/"+path, dst)

    if True:
        cl = TLClassifier()
        
        paths = glob(os.path.join('images/', '*.jpg'))
        for path in paths:
            img = cv2.imread(path)
            status = cl.get_classification( img )
            print cl.tlclasses_d[status], path



