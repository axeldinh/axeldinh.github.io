---
title: "Motion Correction in Cardiac Magnetic Resonance Imaging"
layout: post
date: 2022-10-01 00:00
tag: Computer Vision
image: /assets/projects/master_project/overlay.png
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "Class project for the course 'Machine Learning' at EPFL. We implemented a deep learning model for road segmentation on satellite images."
category: project
author: axeldinh
externalLink: false
icon: /assets/projects/road_segm/icon.png
---

This is a class project for the course 'Machine Learning' at EPFL which took form as an [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation) chalenge. The project was done in groups of 3 students.

The goal was to implement a deep learning model that could detect roads out of satellite images. The dataset used for training and testing the model was provided by the course, it contained
100 images of size 400x400 pixels for training and 50 images of size 608x608 for testing.

![Unet](/assets/projects/road_segm/unet.png)

Using a UNet architecture, along with an appropriate loss function, we were able to achieve an F1 score of 0.908 on the test set, leading us to the 4th place of the leaderboard.

![Segmentation](/assets/projects/road_segm/pipeline.png)
