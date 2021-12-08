# CNN Classifier on Fashion Product Images

Fashion e-commerce accounted for roughly [29.5 percent of the total fashion retail sales in the United States](https://www.statista.com/statistics/281594/share-of-apparel-and-accessories-sales-in-total-us-e-retail-sales/) (2020). Yet, one of the main problems they face is categorizing these apparels, such as clothing and accessories, from the images, especially when the categories provided by the brands are inconsistent. 

In general, image classification poses an exciting computer vision puzzle and has gotten many deep learning researchers' attentions. Building a classification model for fashion product images would be an excellent start to dive into deep neural networks. This project aims to get myself hands-on experience dealing with the imbalanced dataset, building Convolutional Neural Networks (CNN) for image classification with **Keras API** and evaluating the model performances.

##1. Data Source

[**Fashion Product Images**](https://www.kaggle.com/paramaggarwal/fashion-product-images-small) **(545.62 MB, 44k colored images of size 80 x 60 x 3)**

*  **styles.csv** contains 44446 rows and 10 columns.

* **44441** **images** (*.jpg).  Five images (id = 12347, 39401, 39403, 39410, 39425) are missing.

  

## 2. Data Wrangling and Feature Engineering

Here, we focus on data cleaning and feature engineering of **style.csv**. 

1. Refine the **Product Class** for Image Classification: 
   We defined a new categorical feature, `Class`, which combines the three hierarchical labels (`masterCategory`, `subCategory`, `articleType`). The **number of categories in** `Class` is **no larger than 35**. 
2. 
3. 



## 3. EDA



![img](/Users/mendyhsu/Documents_m/Springboard_DataSci/Capstone2/EDA_figs/EDA_fig_FashionCate.pdf)



