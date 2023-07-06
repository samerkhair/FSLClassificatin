# FSLC - Fingerspelling Sign Language Classification 

In this project, we used a classifier trained on deep learning skills to try to translate a video displaying a spelled sign language word to an English word.
Given ten videos with ten different words spelled sign by sign, our classifier predicted almost all of them while failing on only two.

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

#

**After building the Classifier, we are one step closer to achieving our goal. Now we want to interact with the video, learn it, and try to recognize when each sign has been produced in order to accurately split it between each sign and sign.**

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



# Future Works
- Generalize the model to analyze videos – the sign language mostly built from a specific sign for every word, and not fingerspelling it.
- Learning the videos and reduce the instructions.
- Using our network to translate other sign languages besides English.

