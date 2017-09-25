# Stock Predictor
An advanced stock chooser used for investing. Copyright Cameron Knight 2017

## Running
run the data collection tool in stockDataCollection.py before running any of the data testing algorithms 

## Dependancies
* Tensorflow
* Quandle
* Yahoo-finance
* google-trends
* google-finance

## Challanges
### Finding Data Sources
Initialy it seemed clear that a sequence to sequence model trained from bar to bar of the guitar music would make the most 
sense to generate novel music from given input. after training it became appearent that the previous bar of the tab had little
to no effect on the next bar. However, there was an overall pattern to the tablature, so it would go to show that an encoded 
value could have an effective baring to showing overall style of a piece.

## Acomplishments
Learned how to manage and collect large ammounts of data.
## Notes
This project is still in a beta-version. a large portion of the project is focused on data collection and pre-processing.
The project keeps machine learning in mind but is currently unable to preprocess any data using a machine learning algorithm.
