# FSLC - Fingerspelling Sign Language Classification 

In this project, we used a classifier trained on deep learning skills to try to translate a video displaying a spelled sign language word to an English word.
Given ten videos with ten different words spelled sign by sign, our classifier predicted almost all of them while failing on only two signs.

**This is a class project for the Technion's EE046211 - Deep Learning course.**

<div id="header" align="center">
  <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/14.jpg" width="400"/>
</div>

<div id="badges" align="center">
 <a href="https://youtu.be/LSdmXyExC1Q">
    <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/23.png" alt="Youtube Badge" width="75"/>
</a>
</div>

# Introduction
In today's technologically advanced world, the power of artificial intelligence and deep learning has opened up numerous possibilities for improving communication and accessibility. One particular area where this innovation has made a significant impact is in the translation of sign language. Sign language is a rich and expressive form of communication used by deaf and hard-of-hearing individuals, but its interpretation can be challenging for those unfamiliar with its intricate gestures.
Our objective was to create a classifier trained on deep learning skills that could accurately translate a video displaying a spelled sign language word into its corresponding English word.
Our approach involved training a classifier using a vast dataset of sign language images. These images encompassed a wide range of signs, each corresponding to a specific letter in English. By leveraging the inherent capabilities of deep learning, our classifier learned to discern the subtle variations in hand movements, positions, and finger spelling that characterize different sign language letters.
In the following sections, we will delve deeper into the methodology, dataset, and evaluation metrics used in our project. We will also discuss the challenges encountered along the way and the future possibilities for expanding and enhancing our work. By leveraging the power of deep learning, our project aims to empower individuals with hearing impairments and foster inclusive communication in our increasingly interconnected world.

<div id="header" align="center">
  <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/16.jpg" width="800"/>
</div>

# DataSet
Our dataset is [Synthetic ASL Alphabet](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet), which contains 27,000 sign images, 1000 for each sign and 1000 blank, and for each sign the images were divided as 900 for train set and 100 for test set, and we redivided them for 800 train set and 200 train set, totaling 21,600 images for training and 5,400 for test.

<div id="header" align="left">
  <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/4.jpg" width="400"/>
</div>

Each image was resized to 256 × 256 by the transformers to fit the input of our model, and it was also rotated by (-20,20) and horizontally flipped with a probability of 0.5, generalizing it to left and right hand sign images,Furthermore, we created other augmentations with Kornia, such as playing with the color of the photos with colorjitter, flipping all of the images, and changing the perspective they were shown in. These augmentations increased the size of our dataset and helped us generalize it more and more.

**For example:** 

<div id="header" align="center">
    <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/20.jpg" width="500"/>
</div>

# Model Architecture

The network architecture consists of three main parts: convolutional layers, fully connected layers, and the forward function.

**Convolutional Layers:**

The conv_layer module is defined using nn.Sequential(), which allows you to stack multiple layers sequentially.
- The first layer is a convolutional layer with 3 input channels, 32 output channels, and a kernel size of 3x3. It applies convolution on the input image.
- The output of the first convolutional layer is passed through Group Normalization (nn.GroupNorm) with num_groups groups.Group Normalization normalizes the activations
  across the channels. 
- A Parametric Rectified Linear Unit (PReLU) activation function is applied after the Group Normalization.
- The next layer is another convolutional layer with 32 input channels, 64 output channels, and a kernel size of 3x3.
- Another PReLU activation is applied.
- A max-pooling layer with a kernel size of 2x2 and stride 2 is used to downsample the feature map.
- This pattern is repeated for two more sets of convolutional layers, gradually increasing the number of output channels.
- The final output from the conv_layer module is a tensor representing the learned features from the input image.
  
**Fully Connected Layers:**

The fc_layer module is also defined using nn.Sequential().

- The input tensor from the convolutional layers is flattened using x.view(x.size(0), -1), which converts the tensor into a 1D vector.
- A dropout layer with a dropout probability of 0.1 is applied to prevent overfitting.
- The flattened vector is passed through several linear layers with different output sizes, each followed by a non-linear activation function (PReLU or ReLU).
- Dropout layers are applied after some linear layers to further prevent overfitting.
- The final linear layer outputs a tensor of size 27, representing the predicted probabilities for each class.
  
**Forward Function:**

The forward function defines the computation flow of the network.
- The input x is passed through the conv_layer module, which applies the convolutional layers and returns the learned features.
- The output of the conv_layer is flattened.
- The flattened tensor is then passed through the fc_layer module, which applies the fully connected layers to obtain the final predictions.
- The output tensor is returned.

This CNN architecture uses Group Normalization instead of Batch Normalization (commented out lines). Group Normalization divides the channels into groups and computes mean and variance statistics within each group, which helps to normalize the activations in a computationally efficient way.
The network architecture combines convolutional layers for feature extraction with fully connected layers for classification. The PReLU and ReLU activation functions introduce non-linearity to the model, allowing it to learn complex patterns and make predictions.

**Hyperparameters**
- Batch Size:
  - Batch size refers to the number of training examples propagated through the network in a single forward/backward pass.
  - In this case, the batch size is set to 128 using batch_size = 128.
    
- Learning Rate:
  - Learning rate determines the step size at each iteration during the optimization process.
  - In this case, the learning rate is set to 1e-4 (0.0001) using learning_rate = 1e-4.
    
 - Epochs:
   - An epoch represents a complete pass through the entire dataset during training.
   - In this case, the number of epochs is set to 15 using epochs = 15.
   - Training for multiple epochs allows the model to see the entire dataset multiple times, improving its ability to learn patterns and generalize.

- Loss Criterion:
  - The loss criterion defines the objective function used to measure the difference between predicted and true labels during training.
  - In this case, the CrossEntropyLoss criterion is used, which is suitable for multi-class classification problems.
  - CrossEntropyLoss combines a softmax activation and the negative log-likelihood loss, making it suitable for training models to output probability distributions over
    multiple classes.
    
- Optimizer:
  - The optimizer is responsible for updating the model's parameters based on the computed gradients during backpropagation.
  - In this case, the Adam optimizer is used with the model's parameters and the specified learning rate.
  - It is a popular choice due to its efficiency and effectiveness in a wide range of problems.

**We initialized the hyperparameters, including the batch size, learning rate, and number of epochs, based on values obtained from a tutorial that trained a similar model on a different dataset. Although these initial values were not specifically tailored to our dataset, they provided a good starting point for our experiments. It is worth noting that the model achieved a high accuracy even with these initial hyperparameter values, indicating the potential effectiveness of the architecture.**

#

**After building the Classifier, we are one step closer to achieving our goal. Now we want to interact with the video, analyze it, and try to recognize when each sign has been produced in order to accurately split it between each sign and sign.**

# Video Processing 
We utilized a hand motion detector algorithm to track the movement of the hand in our videos. Whenever we detected even the slightest movement, we discarded all the frames captured up to that point and replaced them with a black image. This approach ensured a clear distinction between signs. For each video, we stored its frames in a list. To signify transitions between signs, we inserted a black frame. However, due to imperfections in the hand motion detection technique, we had to further refine our process.

To address this, we developed a mechanism where we analyzed the frames specific to each sign. We tallied the number of frames within a sign, and if it fell below 10 frames, we concluded that it was not a valid sign and discarded it. Our determination was based on observing that each sign typically consisted of at least 20 frames. By applying this criterion, we converted each fingerspelling video into a refined list of frames. This list only contained frames where the sign was clear, with black frames inserted between each sign transition.

By implementing these refinements, we aimed to ensure that our resulting frame lists accurately represented the distinct signs while maintaining clarity during sign transitions.

<div id="header" align="center">
    <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/21.jpg" width="600"/>
</div>

In order for the algorithm to effectively operate, the video footage needed to adhere to specific requirements. The hand motions within the film should be swift and distinct, ensuring clear transitions between signs. To indicate a different sign, it was necessary to change the hand position accordingly.

Moreover, our algorithm was primarily designed to analyze frames where the hand was prominently visible against a plain background. This condition enabled better differentiation and tracking of hand movements, minimizing potential distractions from the surrounding elements.

By adhering to these criteria and constraints, we aimed to optimize the accuracy and reliability of our hand motion detection algorithm, facilitating the extraction of meaningful frames for further analysis and processing

# Prediction Method

After processing the video and obtaining the frame list, it's time to predict the signs using the classifier we developed. By leveraging the classifier and the output frame list, we can make predictions for each sign represented in the video.

Between every two black frames, which indicate sign transitions, we applied our classifier to predict the sign within that specific frame. We stored these individual predictions in another list. Once we finished predicting all the frames for each sign and saved them in the list, we performed an analysis on this list.

To improve accuracy, we implemented a technique to determine the final prediction for each sign. We examined the list of predictions for a given sign and identified the most frequently occurring prediction. This majority prediction was considered the final prediction for that particular sign.

By adopting this approach, we aimed to enhance the accuracy of our overall system. Even if the classifier occasionally provided false predictions for specific frames, the final prediction for a sign would be based on the majority consensus from all the predicted frames within that sign.

# Evaluation and Results
Down below will be displayed some graphs in order to show the model results, and here we show the Evaluations from the last epoch.

<div id="header" align="center">
  <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/99.jpg" width="700"/>
</div>

<div id="header" align="center">
  <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/100.jpg" width="700"/>
  <img src="https://github.com/samerkhair/FSLClassification/blob/main/images/101.jpg" width="700"/>

</div>

# Usage
in order to run our project, you should download all the files from the code directory, and follow the instructions written there.
you have to change some paths in order to make it work.

|  | file | description |
| -------- | -------- | -------- |
|1|[ASLClassifier.ipynb](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet) | this file builds tha classifier   |
|2|[video_prediction.ipynb](https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet)   | loading the classifier and predicting the video   |

  


# Future Works
- Generalize the model to analyze videos – the sign language mostly built from a specific sign for every word, and not fingerspelling it.
- Learning the videos and reduce the instructions.
- Using our network to translate other sign languages besides English.

