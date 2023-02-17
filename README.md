# SES_LSTM
Modified LSTM for Smart Eyedrop System Project

## Platform
- Code had been compiled using Spyder via Anaconda platform. 
- In anaconda prompt, use the following command to install necessary packages:
	- pip install keras 
	- pip install tensorflow
	- pip install sklearn
	- pip install matplotlib
	- pip install pandas
	- pip install flask
	- pip install flask_restful
           
## Models
- Neutral Model: Trained on 100 entries, Tested on 100 entries
- Base Model: Trained on 50 entries, Tested on 50 entries
- Heavy Model: Trained on 250 entries, Tested on 300 entries
> 'combineROC' and 'LSTMReport' are visualization functions. Comment them out if you want the model by itself.

## Connection via Flask
- By default, code is independent of rest of SES. To reconnect, uncomment 'Flask Connection' code block.
