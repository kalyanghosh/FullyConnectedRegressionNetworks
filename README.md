# Cell_Counting_using_FCRNs
This project deals with the task of automated cell counting in microscopy using Fully Constitutional Regression Networks

# TEAM MEMBERS:
1. KALYAN GHOSH
2. BHARGAV RAM
3. VAGEESH

## CONTENTS:
1. ABSTRACT
2. LITERATURE SURVEY
3. INTRODUCTION
4. METHODOLOGY
5. MATHEMATICAL FORMULATIONS
6. MESA DISTANCE
7. ARCHITECTURE
8. DATASETS
9. RESULTS
10. CONCLUSION
11. REFERENCES

## ABSTRACT:
This project deals with the task of automated cell counting in microscopy. The approach we take is to use Convolutional Neural Networks (CNNs) to regress a cell spatial density across the image. This is applicable to situations where traditional single-cell segmentation-based methods do not work well due to cell clumping or overlap. We make the following contributions: (i) We implement two Fully Convolutional Regression Networks (FCRNs) for this task in Keras with Tensorflow backend, we follow the architecture given in [1]; (ii) We fine-tune our model to get more accuracy than what is obtained in [1]; (iii) We compare the results of our architecture which uses density based method to results obtained by an U-Net which uses a segmentation-based approach (iv) We show that FCRNs trained entirely on synthetic data are able to give excellent predictions on real microscopy images. We set a new state-of-the-art performance for cell counting on standard synthetic image benchmarks and, as a side benefit, show the potential of the FCRNs for providing cell detections for overlapping cells.

## LITERATURE SURVEY:
In this project, we implement the architecture proposed in the paper, "Microscopy Cell Counting with Fully
Convolutional Regression Networks" by Zisserman & Weidi at University of Oxford https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf for cell counting and also compare its performance to the traditional segmentation based approach to cell counting using an an U-net convolutional network. 

## INTRODUCTION:
Automatic cell counting can be approached from two directions, one is detection- based counting which requires prior detection or segmentation; the other is based on density estimation without the need for prior object detection or segmentation. Both the methods produce similar error performance.  However, in recent years it has been found that the density-based estimation is a faster method for automatic cell counting than the segmentation method.
In this project, we focus on cell counting in microscopy, but the developed methodology could also be used in other counting applications.

## METHODOLOGY:
In our project, we develop and compare the architecture of two Fully Convolutional Regression Networks (FCRNs) A and B along with UNet which is a Convolutional Network used extensively for Biomedical Image Segmentation. The networks being fully convolutional, they can predict the density map of an input image of arbitrary size. We use this property of fully convolutional networks to improve the efficiency by training end-to-end on image patches. In this project we use synthetic data to train our model. 

![pic1](https://github.com/kalyanghosh/Cell_Counting_using_FCRNs/blob/master/pic1.JPG)

![pic2](https://github.com/kalyanghosh/Cell_Counting_using_FCRNs/blob/master/pic2.JPG)

## MATHEMATICAL FORMULATIONS:
We assume that a set of N training images (pixel grids) I_1,I_2, …., I_N is given. It is also assumed that each pixel p in each image I_i is associated with a real-valued feature vector x_p^iER^K. It is finally assumed that each training image I_i is annotated with a set of 2D points P_i = {P_1, …., P_(C(i))}, where C(i) is the total number of objects annotated by the user.
The density functions in our approaches are real-valued functions over pixel grids, whose integrals over image regions should match the object counts. For a training image I_i , we define the ground truth density function to be a kernel density estimate based on the provided points:

                            ∀p ∈ I_i   F_i^0 (p)= ∑_(P∈P_i)N(p;P,σ^2 I_2x2)                  (1)
In the above equation p denotes a pixel and N(p;P,σ^2 I_2x2) denotes  a normalized 2D Gaussian kernel evaluated at p, with the mean at the user-placed dot P , and an isotropic covariance matrix with σ being a small value (typically, a few pixels). With this definition, the sum of the ground truth density over the entire image will not match the dot count Ci exactly, as dots that lie very close to the image boundary result in their Gaussian probability mass being partly outside the image. This is a natural and desirable behavior for most applications, as in many cases an object that lies partly outside the image boundary should not be counted as a full object, but rather as a fraction of an object.
Given a set of training images together with their ground truth densities, we aim to learn the linear transformation of the feature representation that approximates the density function at each pixel:

                            ∀p ∈ I_i     F_i (p│c)=c^T x_p^i                               (2)

Where cER^K    is the parameter vector of the linear transform that we aim to learn from the training data, and F_i (.│c)  is the estimate of the density function for a particular value of c. The regularized risk framework then suggests choosing c so that it minimizes the sum of the mismatches between the ground truth and the estimated density functions (the loss function) under regularization:

                     c=argmin(c^T c+λ*∑_(i=1)to N D(F_i^0 (.),F_i (.│c)))            (3)

## MESA DISTANCE:

Given an image I, the MESA distance DMESA between two functions F1(p) and F2(p) on the pixel grid is defined as the largest absolute difference between sums of F1(p) and F2(p) over all box subarrays in I:

      D_MESA (F_1,F_2 )=max⁡(max_(B∈B) (∑_(p∈Pa)F_1(p)-F_2(p)),max_(B∈B) (∑_(p∈P)F_2(p)-F_1(p)))   (4)

Here, B is the set of all box subarrays of I. The MESA distance (in fact, a metric) can be regarded as an L distance between combinatorically-long vectors of subarray sums. In the 1D case, it is related to the Kolmogorov-Smirnov distance between probability distributions (in our terminology, the Kolmogorov-Smirnov distance is the maximum of absolute differences over the subarrays with one corner fixed at top-left; thus, the strict subset of B is considered in the Kolmogorov-Smirnov case).

## ARCHITECHTURE:
The ground truth is provided as dot annotations, where each dot corresponds to one cell. For training, the dot annotations are each represented by a Gaussian, and a density surface D (x) is formed by the superposition of these Gaussians. The task is to regress this density surface from the corresponding cell image I (x). This is achieved by training a CNN using the mean square error between the output heat map and the target density surface as the loss function for regression. At inference time, given an input cell image I(x), the CNN then predicts the density heat map D (x). The architecture used in our project is shown in figure 3.
The popular CNN architecture for classification contains convolution-ReLU- pooling. Here, ReLU refers to rectified linear units. Pooling usually refers to max pooling and results in a shrinkage of the feature maps. However, in order to produce density maps that have equal size to the input, we reinterpret the fully connected layers as convolutional layers. The first several layers of our network contains regular convolution-ReLU-pooling, then we undo the spatial reduction by performing upsampling-ReLU- convolution, map the feature maps of dense representation back to the original resolution. During up sampling, we first use bilinear interpolation, followed by convolution kernels that can be learnt during end-to-end training. We present two networks, namely FCRN-A, FCRN-B. 
We only use small kernels of size 3x3 or 5x5 pixels. The number of feature maps in the higher layers is increased to compensate for the loss of spatial information caused by max pooling. In FCRN-A, all of the kernels are of size 3x3 pixels, and three max-pooling are used to aggregate spatial information leading to an effective receptive field of size 38x38 pixels (i.e. the input footprint corresponding to each pixel in the output). FCRN-A provides an efficient way to increase the receptive field, while it contains only about 1.3 million trainable parameters. In contrast, max pooling is used after every two convolutional layers to avoid too much spatial information loss in FCRN-B. In this case, the number of feature maps is increased after every max pooling up to 256, with this number of feature maps then retained for the remaining layers. Comparing with FCRN-A, in FCRN-B we use 5x5 filters in some layers leading to the effective receptive field of size 32x32 pixels. In total, FCRN-B contains about 3.6 million trainable parameters, which is about three times as many as those in FCRN-A

![pic3](https://github.com/kalyanghosh/Cell_Counting_using_FCRNs/blob/master/Architecture.JPG)

![pic4](https://github.com/kalyanghosh/Cell_Counting_using_FCRNs/blob/master/Table.JPG)

In the above table D and K refer to input Dimension and K is the kernel size. For example, 32,3,3 says the input dimension is 32x32 and kernel size is 3,3.MP denotes max-pooling. As can be seen from the architecture FCRN-A has 7 blocks while FCRN-B has 6 blocks. For FCRN-A the block 1 has the inputs 32,3,3 and Max Pooling is 2x2 as the input image has size 100x100 while block 1 has the size 50x50. Similarly, input parameters for block 2 are 64,3,3 and Max-Pooling is 2x2 as block 1 has size 50x50 whereas block 2 has size 25x25. We take appropriate input parameters for other blocks as described in the above rule. For block 6, we first perform up sampling by 2x2 as block 5 has size 25x25 whereas block 6 has size 50x50. We do the same for block 7. A similar approach was taken to code the architecture of FCRN-B.

## DATASETS:

Synthetic Data: We generated 200 fluorescence microscopy cell images, each synthetic image has an average of 174±64 cells. The number of training images was between 8 and 64. After testing on 100 images, we report the mean absolute errors and standard deviations for FCRN-A and FCRN-B. Each image is mapped to a density map first, integrating over the map for a specific region gives the count of that region.

Real data: We evaluated FCRN-A and FCRN-B on two data sets; (1) retinal pigment epithelial (RPE) cell images. The quantitative anatomy of RPE can be important for physiology and pathophysiology of the visual process, especially in evaluating the effects of aging; and (2) Images of precursor T-Cell lymphoblastic lymphoma. Lymphoma is the most common blood cancer, usually occurs when cells of the immune system grow and multiply uncontrollably.
The implementations are based on MatConvNet. Back-propagation and stochastic gradient descent are used for optimization. During training, we cut large images into patches, for instance, we randomly sample 500 small patches of size 100x100 from 500x 500 images. The amount of data for training has been increased dramatically in this way. Each patch is normalized by subtracting its own mean value and then dividing by the standard deviation. The parameters of the convolution kernels are initialized with an orthogonal basis. Then the parameters c are updated by:
                                               Δc(t+1)=β*Δct+(1-β)*α*∂l/∂c                                                  (5)
In the above equation α is the learning rate, and β is the momentum parameter

## RESULTS:

The results obtained from our simulations of FCRN-A, FCRN-B and UNet are shown below. In the tables that follow, the error rate is the absolute difference between the actual count of cells in the image and the predicted count from the 3 architectures.

![pic5](https://github.com/kalyanghosh/Cell_Counting_using_FCRNs/blob/master/Results1.JPG)
![pic6](https://github.com/kalyanghosh/Cell_Counting_using_FCRNs/blob/master/Results2.JPG)

## CONCLUSIONS:

In this project we implemented the FCRN-A and B, which was originally written in MATLAB, in Keras with TensorFlow. We were also able to improve the performance for cell counting from the original paper by fine-tuning the model and using He-normal initialization and also show the potential of the FCRNs for providing cell detections for overlapping cells. It was found that the error rate for the traditional UNet is comparable to that of the FCRN-A and B but the run time of the new FCRN is less than half of that of the UNet. This shows that the FCRNs have a huge potential to be used for cell counting in Bio-Medical Applications. 
