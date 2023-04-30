# Deep image prior with sparsity constraint for limited-angle computed tomography reconstruction

## Brief description of our algorithm in the context of Helsinki Tomography Challenge 2022 (HTC 2022)

Our algorithm is based on the Deep Image Prior [[1]](#1), but we modified it to include sparsity prior information. This was done with a greedy algorithm, the modified matching pursuit that includes a $\ell_0$-norm constraint in the solution [[2]](#2).

* Section 1 describe the real-world problem concerning the HTC 2022.
* Section 2 describes the forward model
* Section 3 describes the installation and basic usage of the algorithm: It may take a few minutes to reconstruct the tomographic image. 

## Authors
* Leonardo Ferreira Alves¹* - leonardo.alves@ufabc.edu.br
* Roberto Gutierrez Beraldo¹ - roberto.gutierrez@ufabc.edu.br
* Ricardo Suyama¹ - ricardo.suyama@ufabc.edu.br
* André Kazuo Takahata¹ - andre.t@ufabc.edu.br
* John Andrew Sims¹ - john.sims@ufabc.edu.br

*Corresponding author

¹Federal University of ABC (Santo André, São Paulo, Brazil) - https://www.ufabc.edu.br/

## 1. HTC 2022 [[3]](#3): Overview, rules, and constraints
Challenge URL: https://www.fips.fi/HTC2022.php

* The challenge consists in reconstructing limited-angle computed tomography (CT) images,  i.e. reducing the number of the total projections to a region,  although it is expected to be a general-purpose (CT reconstruction) algorithm.  
* Several design choices of the algorithm were related to the challenge. 
* After the tomographic image is reconstructed, it is necessary to segment it into two parts, binarizing it: the acrylic disk (1's) and the background, including holes (0's).   
* For more details, please refer to https://www.fips.fi/Helsinki_Tomography_Challenge_2022_v1.pdf.
* The training dataset consists of only 5 sinograms, while the test set contains 21 cases divided in 7 levels of difficulty. They are avaliable at: https://zenodo.org/record/7418878. 
* We know the subsampling of each difficulty group, that is: The information given includes the initial (random) angle, the angular range, and the angular increment. The sinogram size varies for each difficulty group. 
* We do not have information for each difficulty group about the number and size of the holes. 

## 2. Forward problem 
We consider the forward problem, i.e. calculating the sinogram from the tomographic image, as 

$$Ax = y$$

where $x$ is the tomographic image, $A$ is the (linear) forward model, and $y$ is the resulting sinogram [[4]](#4).  

* The cone beam computed tomography considers the detector is flat. As the image is 2D, the cone beam CT can be approximated by the fan beam CT. 
* We use the ODL Pytorch extension (https://github.com/odlgroup/odl) to obtain $A$ not as an explicit matrix, but as an object.
* By the instructions, the submitted algorithm does not need to subsample the test data (as this has already been done). In this way, we only need to create an appropriate $A$ for each difficulty group, given the initial angle and the angular range. 


### Note:
* The above $A$ matrix size is calculated considering an image of 512x512 = 262144 values and  considering 560 (detector points) x [360*2+1] (a projection for each 0.5° angular increment) = 403760 values when including all angles available. Although sparse, $A$ has 262144 x 403760 values considering full-angle CT, so it is often not practical to explicitly calculate $A$.  
 


## 3. Proposed method installation, usage instructions, and examples

### 3.1 Method installation and requirements
* The Python codes are available in this repository, see main.py and the /utils folder.

* We ran our codes using Google Colaboratory (Colab), but it results in a large list of packages (obtained by pip freeze > requirements.txt) and not all of them are necessary.
* It is possible to create an Anaconda environment "by hand" given the packages list. In the following table, there is a small list of the main packages we used (with "import").

| Package | Version |
| ------------- | ------------- |
| Python | 3.10.11 | 
| Numpy | 1.22.4 | 
| Matplotlib | 3.7.1 | 
| Scipy | 1.10.1 | 
| Skimage | 0.19.3 |
| Pillow | 8.4.0 | 
| Torch | 2.0.0+cu118 | 
| ODL | 1.0.0 | 

### 3.2 Usage instructions and example: Running with a callable function from the command line

By the rules of the HTC 2022, it was expected a one-line command: 
* *Your main routine must require three input arguments:*
1. *(string) Folder where the input image files are located*
1. *(string) Folder where the output images must be stored*
1. *(int) Difficulty category number. Values between 1 and 7*
* *Python: The main function must be a callable function from the command line. 

After the setup, it is possible to run our code following these rules. Considering the difficulty group 7: 
* !python main.py 'example/input' 'example/output' 7

See, for instance, the Section "Generating results" from the example notebook [Here](/notebook_example.ipynb).

### 3.3 External codes

* To make our code compatible with PyTorch, it was mostly based on ODL (https://github.com/odlgroup/odl), under Mozilla Public License, version 2.0.
* We also need to mention that we adapted functions from the original DIP article [[4]](#4). Available at https://github.com/DmitryUlyanov/deep-image-prior/, under Apache License 2.0. The particular requisites are shown here: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/README.md

Although these toolboxes have their own requisites, Subsection 3.1 describes the ones we need. 

## References

<a id="1">[1]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior”. International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888 (2020). [Online]. Available at: https://doi.org/10.1007/s11263-020-01303-4

<a id="2">[2]</a> 
Lun Li, Renmin Han, Zhaotian Zhang, Tiande Guo, Zhiyong Liu, and Fa Zhang. 
“Compressed sensing improved iterative reconstruction-reprojection algorithm for electron tomography”. From 15th International Symposium on Bioinformatics Research and Applications (ISBRA’19). Available at: https://doi.org/10.1186/s12859-020-3529-3.

<a id="3">[3]</a> 
Salla Latva-Äijö et al. “Helsinki Tomography Challenge 2022 (HTC 2022).”. Available at: https://www.fips.fi/Helsinki_Tomography_Challenge_2022_v1.pdf. 

<a id="4">[4]</a>
Jari Kaipio and Erkki Somersalon. Linear and nonlinear inverse problems with practical applications. Philadelphia: Society for Industrial and Applied Mathematics (2012).
