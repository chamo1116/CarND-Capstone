## JJ - Traffic Light Classifier

The [Bosch Small Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) looks like a good data set for us to use for the classifier.

Alberto concatenated the separate dataset components into a single 17GB `.tgz` file [here](https://drive.google.com/file/d/1TMb8jBb47Nopy2E5dumr-9mIn2-nwWW_/view).

From the project root, I extracted this file into a `Bosch` directory:

```sh
# if VM:
cd ~/CarND-Capstone
# if Docker:
cd /capstone
# then:
tar xf Bosch_Small_Traffic_Lights_Dataset.tgz -C Bosch
```

I intentionally am not committing these files because they are massive.

I wrote `tl_train/preprocess_bosch_dataset.py` as a utility to preprocess `Bosch/train.yaml` with the following goals:

* Since car camera images are 800 x 600, we'll want to work with the same sized images from the Bosch set.
  * However, the Bosch images are 1280 x 720 - a different aspect ratio (AR).
  * So, we'll perform a crude center crop to give the Bosch images the same AR, making them 960 x 720.
  * Given that, we'll need to filter out any lights that aren't in the resulting cropped image.
* Each image should have a clear label.
  * Some of the images have multiple colors of lights (e.g. green left but red straight).
  * To make things easier for our learner at first, let's filter out such images.

For now, all `tl_train` scripts should be placed in and run from the `tl_train` directory.

### Next steps

We definitely want to take a [transfer learning](http://cs231n.github.io/transfer-learning/), as opposed to training a full CNN from scratch.

[This Keras blog post](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) had some good ideas for how we could apply transfer learning. [This SDC Nanodegree classroom lesson](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/e12c47b6-316e-4a0b-aae5-2f2c5fcd99f5/concepts/10489223-72fa-4393-848b-f882ba3cf7f9) also details different transfer learning approaches.

I'm going to try the "bottleneck" features approach since it seems most applicable to us. [This code](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069) from the Keras blog post looks like a good starting point.

## Original Project README

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases/tag/v1.2).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view?usp=sharing) that was recorded on the Udacity self-driving car (a bag demonstraing the correct predictions in autonomous mode can be found [here](https://drive.google.com/open?id=0B2_h37bMVw3iT0ZEdlF4N01QbHc))
2. Unzip the file
```bash
unzip traffic_light_bag_files.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_files/loop_with_traffic_light.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
