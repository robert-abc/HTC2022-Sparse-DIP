# HTC2022 - Proposed algorithm

By the rules: *Your repository must contain a README.md file with at least the following sections:*
* *Installation instructions, including any requirements.*
* *Python users: Please specify any modules used. If you use Anaconda, please add to the repository an environment.yml file capable of creating an environment than can run your code (instructions). Otherwise, please add a requirements.txt file generated with pip freeze (instructions)*
* *Usage instructions.*
* *Show few examples.*

# Helsinki Tomography Challenge 2022 (HTC 2022): Brief description of our algorithm

* See Section 7 for basic usage of the algorithm: It may take a few minutes to reconstruct the tomographic image. 
* See Section 4 for an overview of the proposed method.
* The other sections describe the challenge rules and detail the (forward) problem model.

## Authors
* André Kazuo Takahata¹ - andre.t@ufabc.edu.br
* John Andrew Sims¹ - john.sims@ufabc.edu.br
* Leonardo Ferreira Alves¹ - leonardo.alves@ufabc.edu.br
* Ricardo Suyama¹ - ricardo.suyama@ufabc.edu.br
* Roberto Gutierrez Beraldo¹ - roberto.gutierrez@ufabc.edu.br

¹Federal University of ABC (Santo André, São Paulo, Brazil) - https://www.ufabc.edu.br/

## 1. HTC 2022 [[1]](#1): Overview, rules, and constraints
Challenge URL: https://www.fips.fi/HTC2022.php

* The challenge consists in reconstructing limited-angle computed tomography (CT) images,  i.e. reducing the number of the total projections to a region,  although it is expected to be a general-purpose (CT reconstruction) algorithm.  
* An example of limited angle data is illustrated below (Imagem from [[2]](#2)).
<p align="center">
<img src="https://github.com/robert-abc/HTC2022/blob/main/figures/limited.png" width="700">
</p>

* After the tomographic image is reconstructed, it is necessary to segment it into two parts, binarizing it: the acrylic disk (1's) and the background, including holes (0's).   
* For more details, please refer to https://www.fips.fi/Helsinki_Tomography_Challenge_2022_v1.pdf.



### 1.1 Dataset from the HTC2022  (https://zenodo.org/record/6984868)
There are only 5 examples of sinograms and their respective filtered-backprojections (FBP) reconstructions. One is a solid disk and the others have holes, which are shown below:
<p align="center">
<img src="https://github.com/robert-abc/HTC2022/blob/main/figures/extraSamples_A.png" width="200">  <img src="https://github.com/robert-abc/HTC2022/blob/main/figures/extraSamples_B.png" width="200">   <img src="https://github.com/robert-abc/HTC2022/blob/main/figures/extraSamples_C.png" width="200">   <img src="https://github.com/robert-abc/HTC2022/blob/main/figures/extraSamples_D.png" width="200">  
</p>

### 1.2 Challenge constraints:
By the rules: *The actual challenge data consists of 21 phantoms, arranged into seven groups of gradually increasing difficulty, with each level containing three different phantoms, labeled A, B, and C. (...) that will be made public by the end of the competition*
* We know the subsampling of each difficulty group, that is: The information given includes the initial (random) angle, the angular range, and the angular increment. 
* The sinogram size varies for each difficulty group. 
* We do not have examples for each difficulty group about the number and size of the holes (only one example of A, B, and C). 

## 2. Forward problem 
We consider the forward problem, i.e. calculating the sinogram from the tomographic image, as 

$$Ax = y$$

where $x$ is the tomographic image, $A$ is the (linear) forward model, and $y$ is the resulting sinogram [[3]](#3).  

* The cone beam computed tomography considers the detector is flat. As the image is 2D, the cone beam CT can be approximated by the fan beam CT. 
* We use the TorchRadon Pytorch extension [[4]](#4) (https://github.com/matteo-ronchetti/torch-radon) to obtain $A$ not as an explicit matrix, but as an object that can multiply a vector, such as $Ax$ or $A^Tx$.  
* By the instructions, the submitted algorithm does not need to subsample the test data (as this has already been done). In this way, we only need to create an appropriate $A$ for each difficulty group, given the initial angle and the angular range. 


### Note:
* The above $A$ matrix size is calculated considering an image of 512x512 = 262144 values and  considering 560 (detector points) x [360*2+1] (a projection for each 0.5° angular increment) = 403760 values. Although sparse, $A$ has 262144 x 403760 values considering full-angle CT, so it is often not practical to explicitly calculate $A$.  
 
## 3. Review of DIP-based CT algorithms 

The main idea was based on the Deep Image Prior [[5]](#5), but we modificated it to include other prior information. 

### 3.1 Original DIP for general image processing  [[5]](#5)

* Input: Sinogram
* Output: Tomographic image

The DIP consists of a deep generator network $f_{\theta}(z)$, a parametric function where the generator weights $\theta$ are randomly initialized and $z$ is a random vector.  

During the traning phase, the weights are adjusted to map $f_{\theta}(z)$ to the tomographic image $x$, as the equation below includes the fan beam CT forward model $A$: 

$$ \hat{\theta} = \arg\underset{\theta}{\min} E (A f_{\theta}(z), y) $$

where $\theta$  are the weights of the generator network $f$ after fitting to the sinogram $y$, the superscript ^ denotes an estimation, and $E$ is the loss function.  

After this, the partial reconstructed image $\hat{x_1}$ is generated by the network using  

$$\hat{x_{\theta}} = f_{\hat{\theta}}(z)$$

The code is available at https://github.com/DmitryUlyanov/deep-image-prior.


### 3.2 DIP for limited-angle CT reconstruction: overview [[6]](#6)

* Input: Sinogram
* Output: Tomographic image

In [[6]](#6), the authors use the anisotropic TV and $\ell_1$-norm in the fidelity term, that is, 

$$ \hat{\theta} = \arg\underset{\theta}{\min} ( ||A f_{\theta}(z) - y||_1 + \lambda ||\nabla x||_1) $$

But, instead of using only convolutional layers in $f_{\theta}$ as the original DIP, they use a two cascaded neural network: It begins with a fully connected (neural) network (FCNN) before the convolutional layers, in a way that this FCNN obtains the tomographic image given the sinogram (reconstruction problem) and the following convolutional layers of the DIP improves the result. 

They also use the ADMM to solve this functional. No code was publicly found.

### 3.3 Compressed sensing improved iterative reconstruction-reprojection (CSIIRR) for limited-angle electron tomography image reconstruction: overview [[7]](#7)

In [[7]](#7), the authors developed an algorithm called CSIIRR to reconstruct electron tomographic images, that inherently presents limited-angle data. After preprocessing the data, it followed repeatedly:
1. Reprojecting the reconstruction of the last iteration to estimate the unknown projections;
1. Reconstructing the tomographic image using modified matching pursuit (MMP), an greedy algorithm that includes a $\ell_0$-norm constraint in the solution. The MMP algorithm can be found in Algorithm 1 of the same work:

$$ min ||y||_0 $$

 $$ \text{s.t.} \quad \chi_{\Omega} A f_{\theta}(z) = y_{known},  $$

where $\chi_{\Omega}$ is a characteristic function of the set $\Omega$, with values equal to $1$ if the angle/projection is known and $0$ if it is unknown.


## 4. Proposed method:

* Input: Limited-angle sinogram, solid disk tomographic image
* Output: Reconstructed tomographic image

In this work, we propose to use the DIP and CSIIRR together in the same algorithm. 

### 4.1 First step: Reconstructing only with DIP

The first step consists in estimating the tomographic image using only the DIP:

$$ \hat{\theta} = \arg\underset{\theta}{\min} ( ||A f_{\theta}(z) - y||_1 + \lambda_1||\theta||_2) $$

$$\hat{x_{\theta}} = f_{\hat{\theta}}(z)$$

Originally, DIP considers a random tensor as the only input. In our work, the input is the tomographic image of the solid disk from the HTC training dataset with an additive noise. We considered 950 iterations in this step. 


### 4.2 Second step: Reconstructing with DIP and CSIIRR together

The second step consists in obtaining a regularization term (substeps i-iv from below) and adding it to the DIP functional (substep v from below) given by

$$\hat{\theta} = \arg\underset{\theta}{\min} ( ||A f_{\theta}(z) - y||_1 + \lambda_1||\theta||_2 + \lambda_2|| (1-\chi_{\Omega}) A x^* - (1-\chi_{\Omega}) A f_{\theta}(z) ||_1) $$ 

where $x^*$ is the obtained prior. 

Then, the reconstructed tomography is given by

$$\hat{x_{\theta}} = f_{\hat{\theta}}(z)$$

* **Outer loop** (repeat three times)
   1. Such as in CSIIRR, estimating the unknown projections (subsection 3.3.1). In the first iteration, the input is the output from the first step. 
   1. **First Inner loop**: Such as in CSIIRR, reconstructing via MMP (subsection 3.3.2). We will call this result $x^{MMP}$. Note that it is an iterative algorithm with inner iterations defined by a stopping criterion that depends on a user-defined tolerance. 
   1. Denoising the result of *ii* with non-local means. We will call this result as $x_{den}^{MMP}$. 
   1. Using the morphological operation closing (erosion of the dilation) in the result of *iii*. We will call this result simply as $x^*$ and will include it in a regularization term in the following DIP reconstruction of *iv*.
   1. **Second inner loop**: Reconstructing the tomographic image using the DIP using the solid disk + noise as the DIP input and $x^*$ in a regularization term. Note that it is an iterative algorithm with inner iterations defined by a fixed number of iterations.
   1. Repeat

## 5. Reconstruction postprocessing: Segmentation 

By the instructions: *The competitors do not need to follow the above segmentation procedure, and are encouraged to explore various segmentation techniques for the limited-angle reconstructions.* The segmentation method proposed by the challenge organizers is available at  https://www.fips.fi/HTCdata.php and is mostly based on the Otsu's segmentation method.

Instead of using this method, we performed the following steps on the tomography obtained in Section 4:

* Preprocessing the output by denoising it with nonlocal means
* Calculating the threshold with the Otsu's method
* Adjusting a sigmoid to the denoised image, considering the threshold 
* Creating a checkerboard level set with binary values
* Using the Chan-Vese segmentation algorithm considering the checkerboard level set as the starting level set.

For more information, see segment_reconstruction(rec_img) at /utils/tools.py 

 
## 6. Reconstruction assessment method: Confusion matrix 

* The organizers will use the following code for evaluating the reconstructions: https://www.fips.fi/main_confMatrix.py
* For more details, please refer to https://www.fips.fi/Helsinki_Tomography_Challenge_2022_v1.pdf.

## 7. Proposed method installation, usage instructions, and examples

* Note that we will make this repository public until November 30, 2022

### 7.1 Method installation and requirements
* The Python codes are available in this repository, see main.py.

* We ran our codes using Google Colaboratory (Colab), but it results in a big list of packages (obtained by pip freeze > requirements.txt) and not all of them are necessary.
* It is possible to create a anaconda environment "by hand" given the packages list. In the following table, there is a small list of the main packages we used (with "import").

| Package | Version |
| ------------- | ------------- |
| Python | 3.7.12 | 
| Numpy | 1.21.6 | 
| Matplotlib | 3.5.3 | 
| Scipy | 1.7.3 | 
| Skimage | 0.18.3 |
| Pillow | 9.3.0 | 
| Torch | 1.6.0+cu101 | 
| TorchRadon | 1.0.0 | 

### 7.2 Usage instructions: Running with a callable function from the command line

By the rules, it was expected an one-line command: 
* *Your main routine must require three input arguments:*
1. *(string) Folder where the input image files are located*
1. *(string) Folder where the output images must be stored*
1. *(int) Difficulty category number. Values between 1 and 7*
* *Python: The main function must be a callable function from the command line. To achieve this you can use sys.argv or argparse module. Example calling the function:*
* *$ python3 main.py path/to/input/files path/to/output/files 3*

After the setup, it is possible to run our code by:

See, for instance, the Section "Generating results" from the exame notebook [Here](/notebook_example.ipynb).



### 7.3. Alternative: Running with Google COLAB

We created a notebook to run the code using Google Colab. The resulting Jupyter Notebook can be found in [Here](/notebook_example.ipynb).

There are some instructions in the notebook itself. Here is a general view for it, which considers that this repository is stil private:
* First, we clone the private git repository.    
* It's not recommended to upload the files directly into the Colab with a free account because of running time limitations. So, the HTC (test) dataset can be uploaded to a google drive account, linking it to the Google Colab via "mount drive" 
* Google Colab will ask for a verification code and then it is possible to access Google Drive directly from the Google Colab.
* After this, it is possible to execute the rest of the code.
* [Link](https://colab.research.google.com/drive/1A3THOEL-haZPkHg9il36SQroA2L_3Box?usp=sharing)

### 7.4 External codes

* To make our code compatible with PyTorch, it was mostly based on Torch Radon (https://torch-radon.readthedocs.io/en/latest/).
* We also need to mention that we adapted functions from the original DIP article [[5]](#5). Available at https://github.com/DmitryUlyanov/deep-image-prior/, under Apache License 2.0. The particular requisites are shown here: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/README.md

* Although these toolboxes have their own requisites, Subsection 7.1 describes the ones we need. 


## References
<a id="1">[1]</a> 
Salla Latva-Äijö et al. “Helsinki Tomography Challenge 2022 (HTC 2022).”. Available at: https://www.fips.fi/Helsinki_Tomography_Challenge_2022_v1.pdf. 

<a id="3">[2]</a>
Jennifer Mueller and Samuli Siltanen. Statistical and Computational Inverse Problems. New York: Springer-Verlag (2005). 

<a id="3">[3]</a>
Jari Kaipio and Erkki Somersalon. Linear and nonlinear inverse problems with practical applications. Philadelphia: Society for Industrial and Applied Mathematics (2012). 

<a id="4">[4]</a> 
Matteo Ronchetti. 
"Torchradon: Fast differentiable routines for computed tomography". arXiv preprint arXiv:2009.14788, 2020. Available at: https://github.com/matteo-ronchetti/torch-radon 

<a id="5">[5]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior”. International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888 (2020). [Online]. Available at: https://doi.org/10.1007/s11263-020-01303-4

<a id="6">[6]</a> 
Semih Barutcu, Selin Aslan, Aggelos K. Katsaggelos and Doğa Gürsoy.
“Limited‑angle computed tomography with deep image and physics priors”. Scientific Reports, vol. 11, 17740 (2021). Available at: https://doi.org/10.1038/s41598-021-97226-2

<a id="7">[7]</a> 
Lun Li, Renmin Han, Zhaotian Zhang, Tiande Guo, Zhiyong Liu and Fa Zhang. 
“Compressed sensing improved iterative reconstruction-reprojection algorithm for electron tomography”. From 15th International Symposium on Bioinformatics Research and Applications (ISBRA’19). Available at: https://doi.org/10.1186/s12859-020-3529-3.
