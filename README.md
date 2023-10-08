# transistorcurrent_value
 a model that predicts the transistor current value based on some parameters.

This repository contains a deep learning model to predict the transistor current (id(uA)) based on various parameters.

Prerequisites:
Python 3.x
TensorFlow 2.x
scikit-learn
pandas
You can install the required packages using:

Copy code
pip install tensorflow scikit-learn pandas
Dataset:
Download the dataset from this link and place it in the root directory of this repository. Rename the dataset to transistor_data.csv.

Running the Model:
Clone this repository:
bash
Copy code
git clone <repository_url>
Navigate to the repository directory:
bash
Copy code
cd <repository_name>
Run the model:
Copy code
python transistor_model.py
Model Details:
The model is implemented using TensorFlow and Keras. It uses a deep neural network with several layers, including dense layers, dropout for regularization, and batch normalization. The model is trained to minimize the mean absolute percentage error (MAPE) for predicting the transistor current (id(uA)).

For more details on the transistor behavior and the underlying physics, refer to this lecture.

