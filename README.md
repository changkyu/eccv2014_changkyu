# Monocular Multiview Object Tracking with 3D Aspect Parts

Yu Xiang*, Changkyu Song*, Roozbeh Mottaghi, and Silvio Savarese (*equal contribution), Monocular Multiview Object Tracking with 3D Aspect Parts, in European Conference on Computer Vision 2014 (ECCV 2014) [[PDF]](https://drive.google.com/file/d/0B_bATRP0GEOjcllWNlBDNnA0NDA/view?usp=sharing) [[Video]](https://www.youtube.com/watch?v=3k0IWm9V2AU) [[Project Page]](https://sites.google.com/site/changkyusong86/research/eccv2014)

In this work, we focus on the problem of tracking objects under significant viewpoint variations, which poses a big challenge to traditional object tracking methods. We propose a novel method to track an object and estimate its continuous pose and part locations under severe viewpoint change. In order to handle the change in topological appearance introduced by viewpoint transformations, we represent objects with 3D aspect parts [1] and model the relationship between viewpoint and 3D aspect parts in a part-based particle filtering framework. Moreover, we show that instance-level online-learned part appearance can be incorporated into our model, which makes it more robust in difficult scenarios with occlusions. Experiments are conducted on a new dataset of challenging YouTube videos and a subset of the KITTI dataset [2] that include significant viewpoint variations, as well as a standard sequence for car tracking. We demonstrate that our method is able to track the 3D aspect parts and the viewpoint of objects accurately despite significant changes in viewpoint. (a) An example output of our tracking framework. Our multiview tracker provides the estimates for continuous pose and 3D aspect parts of the object. (b) An example of the 3D aspect part representation of a 3D object (car) and the projections of the object from different viewpoints.

# Dataset

Multiview Tracking Dataset ~650MB (uploaded on stanford) [[Download]](http://www.google.com/url?q=http%3A%2F%2Fcvgl.stanford.edu%2Fdata%2Fmultiview_tracking_dataset.zip&sa=D&sntz=1&usg=AFQjCNF3QDe5SRqVfM6g54d3--PzbWxLaA)

# Author
* [Changkyu Song](https://sites.google.com/site/changkyusong86) (changkyusong86@gmail.com)
