# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[left]: ./examples/left.jpg "Left Camera"
[right]: ./examples/right.jpg "Right Camera"
[center]: ./examples/center.jpg "Center Camera"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 with a captured video of driving in autonomous mode for the entire lap of track1

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```python drive.py model.h5```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a series of:

- preprocessing: normalization and cropping
- a series of convolution and maxpooling, to reduce the dimensionality of data
- a flattening layer
- a series of dense layers with dropout

A RELU activation is used throughou.


#### 2. Attempts to reduce overfitting in the model

The model uses maxpooling and dropout layers in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 85). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 83).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving,
and additional short segments to address problematic areas.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have the network express a relatively simple rule: when getting close to a curb, steer away from it.

This suggested choosing a model that could extract relatively staightforward large-scale features.

It seemed intuitive that the model could be simpler than what was required for traffic signs or even handwritten digits, but should be able to handle the relatively large image size.

My approach was to start with a simple network and then vary the complexity.

The challenge proved to be primarily in choosing the right training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 49-83) consisted of a convolution neural network with the following layers and layer sizes:

- normalization using a Lambda layer
- a cropping layer to cut off the skyline and car bumper
- a series of 3 pairs of convolutional (32x3x3) and maxpooling (2x2) layers, with a RELU activation
- a flattening layer
- a series of dense layers (sizes 50, 10, 1) interspersed with a dropout layers (rate=0.4), with RELU activations (except for the output)


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][center]

I've soon realized that much of my driving data had zero steering angle, so I decided to follow the recommendation of using the left and right cameras with adjusted steering angle. With this, I've committed to the approach of "repelling" curbs -- as soon as vehicle starts approaching the curb, the driver should steer away from the curb.

After much experimentation with different mix of data, both raw and augmented, I settled on only using the left and right camera images.


![left][left]
![right][right]


To augment the data sat, I have also experimented with flipping the images and angles thinking that this would improve generalized behavior. In the end, I've opted out of using flipped data.

Additionally, I've drove the track manually in opposite direction, but abandoned that dataset early on, as it didn't seem to benefit significantly, while effectively doubling the amount of data and therefore the training time.

After the collection process, I had 4836 number of data points. I preprocessed this data by simple normalization and cropping.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the fact that both training and validation errors reached minimum at 10th epoch.

I used an adam optimizer so that manually training the learning rate wasn't necessary.
