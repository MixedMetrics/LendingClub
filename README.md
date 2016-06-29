# LendingClub
Machine learning tools to predict the best loans available on Lending Club.  Uses xgboost in python.  

This particular program take the file of notes issued in 2015 from Lending Club, and trains a predictor for the current notes that are in funding.  

The target I'm using is the balance after 5 years, if you bought only this loan and reinvested the payments continously. 

To use this: 
1)  Go to Lending club page (follow Statistics tabs) and download the data : there's several files from different year.  There's different files for different years.   In this example I used only the 2015 data.  Save as "train.csv" 

Note that the grades of the loans offered today are much different.  The low grade loans produce a lot of variance. For this reason I only use the A,B,C grade loans in the training.  

2) Download the current retail loans, save as "retail.csv"


The training takes a few minutes, so if you plan to do this often, it's probably best to pickle the booster after you've trained it.  
