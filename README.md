# Detection for anomalies in fruits and vegetables

## Overview
The Spoilage Detection System is an AI-powered application designed to automate the detection of food spoilage, ensuring compliance with food safety regulations. By utilizing computer vision and machine learning techniques, this system helps food companies enhance their quality control processes, minimize waste, and improve operational efficiency.

## Features
- **Automated Spoilage Detection:** Quickly classify fruits and vegetables as fresh or spoiled using a combination of OpenCV and a Convolutional Neural Network (CNN).
- **Real-Time Reporting:** Generate immediate classification reports for quality control.
- **Arduino Integration:** An Arduino system that activates a bulb to indicate spoilage—lit when a vegetable is spoiled, and off when fresh.
- **Scalable Architecture:** Designed to handle large-scale inputs for food processing facilities and distribution centers.
- **Integration Capabilities:** Seamlessly connect with existing inventory management and regulatory compliance systems.

## Problem Statement
Current food safety practices are often manual, leading to inefficiencies and errors. The need for a reliable and automated solution is critical to maintaining quality assurance and adhering to regulatory standards in the food supply chain.

## Data Sources
- **Images:** Fresh and spoiled fruits/vegetables categorized for training and testing.
- **Metadata:** Time of collection, type of produce, temperature conditions (if applicable).

## Technologies Used
- **Programming Languages:** Python
- **Libraries:** OpenCV, TensorFlow, Keras, NumPy
- **Machine Learning Framework:** Convolutional Neural Networks (CNN)
- **Hardware:** Arduino for spoilage indication
- **Data Handling:** ImageDataGenerator for preprocessing and augmentation



