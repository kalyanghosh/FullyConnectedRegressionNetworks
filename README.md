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
For this project, we were influenced by the paper by Zisserman & Weidi at University of Oxford https://www.robots.ox.ac.uk/~vgg/publications/2015/Xie15/weidi15.pdf

## INTRODUCTION:
Automatic cell counting can be approached from two directions, one is detection- based counting which requires prior detection or segmentation; the other is based on density estimation without the need for prior object detection or segmentation. Both the methods produce similar error performance.  However, in recent years it has been found that the density-based estimation is a faster method for automatic cell counting than the segmentation method.
In this project, we focus on cell counting in microscopy, but the developed methodology could also be used in other counting applications.

## METHODOLOGY:
In our project, we develop and compare the architecture of two Fully Convolutional Regression Networks (FCRNs) A and B along with UNet which is a Convolutional Network used extensively for Biomedical Image Segmentation. The networks being fully convolutional, they can predict the density map of an input image of arbitrary size. We use this property of fully convolutional networks to improve the efficiency by training end-to-end on image patches. In this project we use synthetic data to train our model. 
![pic1](https://github.com/kalyanghosh/Cell_Counting_using_FCRNs/blob/master/pic1.JPG)

