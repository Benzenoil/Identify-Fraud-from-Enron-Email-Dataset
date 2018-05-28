# Enron-Dataset-Project
This is the final project for Udacity Machine Learning Course  
Link to the Udacity Project repo: https://github.com/udacity/ud120-projects  
Link to the Enron Dataset Source: https://www.cs.cmu.edu/~enron/

## Overview
In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. In this project, you will play detective, and put your new skills to use by building a person of interest identifier based on financial and email data made public as a result of the Enron scandal. To assist you in your detective work, we've combined this data with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.

## Goal
I need to build machine learning model to identify the POI(Person of Interest) in the dataset.

## About the File in this repo
poi_id.py: Main program to identify the POI, comparing the performance with different machine learning way.
<b>If you want to run this file in your PC you have to 
  clone or download the project repo showed at top of the README, then replace the file in the final_project floder. </b> 
  Also, the Enron Dataset is necessarily.  
tester.py: Tester file exist in Udacity Project repo which can show the performance of the selected classifier.  
[my_classifier, my_dataset, my_feature_list]: Dataset which tester.py used it to evalutate the classifier.  
refer.txt: Reference which I referred in my project.  
Report_zh_JunWang_ver1.1.pdf: The report of the project in Chinese.  

## Discussion and Conclusion
In this project, I finally chose the GaussianNB to identify the POI in the enron dataset because of the most balanced performance.Here is the Performance:

First Header | Second Header
------------ | -------------
Accuracy | 0.84300
Precision | 0.48581
Recall | 0.35100
F1-score | 0.40755
F2-score | 0.37163
Total predictions | 13000
