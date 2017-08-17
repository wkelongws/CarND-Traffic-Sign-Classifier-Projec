# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[train_distribution]: ./output_images/train_distribution.png "train_distribution"
[ImageVis_eachClass]: ./output_images/ImageVis_eachClass.png "ImageVis_eachClass"
[gray_sharp_effect]: ./output_images/gray_sharp_effect.png "gray_sharp_effect"
[jitter_effect]: ./output_images/jitter_effect.png "jitter_effect"
[orginal_distribution]: ./output_images/orginal_distribution.png "orginal_distribution"
[val_distribution]: ./output_images/val_distribution.png "val_distribution"
[0_pred_dist]: ./output_images/0_pred_dist.png "0_pred_dist"
[1_pred_dist]: ./output_images/1_pred_dist.png "1_pred_dist"
[2_pred_dist]: ./output_images/2_pred_dist.png "2_pred_dist"
[3_pred_dist]: ./output_images/3_pred_dist.png "3_pred_dist"
[5_pred_dist]: ./output_images/5_pred_dist.png "5_pred_dist"

[Children crossing]: ./output_images/Childrencrossing.jpg "Children crossing"
[Priority road]: ./output_images/Priorityroad.jpg "Priority road"
[Right-of-way at the next intersection]: ./output_images/Right-of-wayatthenextintersection.jpg "Right-of-way at the next intersection"
[Roundaboutmandatory]: ./output_images/Roundaboutmandatory.jpg "Roundaboutmandatory"
[Turnrightahead]: ./output_images/Turnrightahead.jpeg "Turnrightahead"


---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

After class-balancing and data augmentation by jittering, I randomly select 80% as training set and 20% as validation set. The test set is preseved and not-jittered from the beginning. The final numbers feed to the model are shown below:

* The size of training set is 77434
* The size of the validation set is 19359
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an visualiztion of one random selected image from each of the 43 classes:

![alt text][ImageVis_eachClass]

The image distribution of the 43 classes in the provided training dataset is unbalanced as shown below:

![alt text][orginal_distribution]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I preprocessed the data in 5 steps:

1. convert to grayscale

I believe the shape should not be the main factor to detect a traffic sign rather than color. If the color of a traffic sign fade, one should still be able to tell what it is. And using gray color should also help to make the model more transferable. For example, in german the "keep right" sign has a blue background but in US it may just be black and white. So in such case, the model should still work. Also in Pierre Sermanet and Yann LeCun's paper, they reported that grayscale images perform better than color images. I tried both BRG2GRAY and BGR2YUV then only use Y channel, both ways perform similar. And here I used BGR2YUV then use only Y channel.

2. Sharp the image by histogram equalization

Sometimes because of the low luminance the image has every low contrast. Use histogram equalization and increase the contrast and make the image more readable.

Here is an example of a traffic sign image before and after grayscaling and sharpening:

![alt text][gray_sharp_effect]

3. z-core the image by substracting mean and devided standard deviation

I personally feel like it is always good to normalize the data.

4. generate more data

From the frequency bar chart in the visualization section we can see the data is unbalanced across classes. So in order to balance the data, I generate additional data samples by jittering the original image. The jittering is conducted by applying affine transformations (such as rotations, translations and shearing). Affine transformations are transformations where the parallel lines before transformation remain parallel after transformation.

Here is an example to show the generated data from jittering:

![alt text][jitter_effect]

5. split into train and validation

80% traning set with about 77524 samples, 20% validation set with about 19381 samples, and 12630 test samples. Because of the random process in the jitter function, the number of training samples and the number of validation samples may vary a little.

After jittering and splitting, the final class distribution of training dataset is shown below:

![alt text][train_distribution]

The final class distribution of validation dataset is shown below:

![alt text][val_distribution]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I borrowed some ideas from both Pierre Sermanet and Yann LeCun's paper and LeNet. My network has 3 covolutinonal layers and 3 fully connected layers.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x3x32		|
| RELU					|												|
| concat*		      	| (see description below)	 			    	|
| Fully connected		| input 982 output 120        					|
| Fully connected		| input 120 output 84        					|
| Fully connected		| input 84 output 43        					|
| Softmax				| 	        									|

 
*here I concatinate output from all 1st, 2nd and 3rd layer and input them to the next fully connected layer. This is the idea borrowed from Pierre Sermanet and Yann LeCun's paper. To reduce the feature size, output from 1st layer are maxpulled again. So in total, 7x7x6 = 294 features from 1st layer, 5x5x16 = 400 features from 2nd layer and 3x3x32 = 288 features from 3rd layer are concatinated together.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The final parameters I used:

* Adam optimizer,

* batch size = 512 

* learning rate = 0.001 

* epochs = 500

It takes about 16 min to train using 4 Geforce GTX TITAN X

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I have tried:

* add a l2 normalizer to the objective function

* less epoches

* different batch sizes

* use dropout

It has not much difference but I decided to add a l2 normalizer to the objective function Different bath sizes don't impact the accuracy much The more epoches the better but obviously more time needed Dropout will decrease the validation accuracy with the same training epoches a lot. I may take a much longer time to train the network with dropout to reach the same accuracy. Since I feel there is not much concerns of overfitting in this problem I decided not to use dropout.

My final model results were:

* training set accuracy of 0.995

* validation set accuracy of 0.932

* test set accuracy of 0.933

The training code and result can be found in cell 'training' (In [17]) in Traffic_Sign_Classifier.ipynb
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][Children crossing] ![alt text][Priority road] ![alt text][Right-of-way at the next intersection] 
![alt text][Roundaboutmandatory] ![alt text][Turnrightahead]

The first image might be difficult to classify because the image contains another small sign at the bottom.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        				| Prediction	        				| 
|:-------------------------------------:|:-------------------------------------:| 
| Children crossing    					| dangerous curve to the right  		| 
| Priority road    						| Priority road							|
| Right-of-way at the next intersection	| Right-of-way at the next intersection	|
| Roundaboutmandatory      				| Roundaboutmandatory					|
| Turnrightahead						| Turnrightahead      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

The model is pretty sure about the prediction for all the images exept the first one which it predicts wrong.

The prediction confidence distribution of the first image is shown below:
The right class "Children crossing" comes at the second highest likelihood.

![alt text][0_pred_dist]

The prediction confidence distribution of the other four images is shown below:

![alt text][1_pred_dist]

![alt text][2_pred_dist]

![alt text][3_pred_dist]

![alt text][5_pred_dist]



