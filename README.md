# SMAD-on-fire
  Based on the detection of fire in videos, binocular stereo matching technology is used to determine the location of fire sources .
  
  # Background
  With the development of modern urban buildings in the direction of large space and high level, the fire source identification and location system that can precisely locate the flame position has a strong practical significance for the prevention and control of fire losses. 
  
  Aiming at the shortcomings of traditional fire location technology such as infrared detection and gas detection, based on the pre recognition of fire images, binocular stereo matching technology is used to determine the disparity of images, and then the location of fire sources is calculated by affine geometry. Considering the particularity of the flame image, looking for a fast and accurate fire recognition algorithm, combined with the timeliness and accuracy of the fire location, we design a binocular stereo vision system which can recognize and locate the fire flame position timely.
  
  # Our Work
The main research work is as follows:

 1.Based on the analysis of the image characteristics of the actual fire scenes, aiming at the possible interference factors in the reality, we discuss how to identify the fire phenomena accurately, and design the process of identifying the fire which use multiple criteria.

2. In the part of fire recognition, the segmentation of foreground image and the feature extraction of flame are realized by using color model and morphological processing. With the extracted spectrum, round degree and other characteristics are used to determine whether a fire is happening. The feasibility of the recognition algorithm is verified by the experiment of fire identification and the analysis of the related results.

3. In the fire location part, we propose an optimized SURF algorithm based on image filling. Through feature extraction and filling of the processed pre segmentation image matching, we candidate matching point. Through screening matching points and the space affine of better matching points, the actual space coordinates of the fire source can be obtained.

4. We design the system architecture which can be put into practical fire source identification and positioning system, and make a prospect for further optimization.
