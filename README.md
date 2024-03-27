# plant_disease_classsifier



Plant Disease Classifier Using Deep Learning

Overview:
This project aims to create a plant disease classifier using deep learning techniques, specifically Convolutional Neural Networks (CNNs) implemented using TensorFlow and Keras. The dataset used for training and testing consists of images of various plant species affected by different diseases. The classification model will be built and deployed using Streamlit, allowing for a user-friendly interface to classify plant diseases from input images.

Requirements:
Python 3.x
TensorFlow
Keras
Streamlit
NumPy
Matplotlib
Pandas
OpenCV (for image processing)
Installation:
Clone the repository:

##Install dependencies:

Copy code
pip install -r requirements.txt
Download the dataset and place it in the appropriate directory (/data).

Usage:
Navigate to the project directory:

bash
Copy code
cd plant_disease_classifier
Train the model:

Copy code
python train_model.py
After training, run the Streamlit app:

Copy code
streamlit run app.py
Access the application via the provided local URL (typically http://localhost:8501).

Upload an image of a plant leaf with potential disease for classification.

Get the predicted disease category along with the probability score.

Model Architecture:
The model architecture consists of a series of convolutional and pooling layers followed by fully connected layers for classification. Transfer learning with pre-trained models such as VGG16, ResNet, or Inception can also be explored for better performance.

Dataset:
The dataset used for training and testing the model contains images of various plant species affected by different diseases. Ensure proper data preprocessing and augmentation techniques are applied to handle class imbalances and improve model generalization.

Contribution:
Contributions to this project are welcome. If you encounter any bugs, have feature requests, or want to contribute enhancements, please submit an issue or pull request.

The Dataset can be downloaded from [Click Here](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
Install all dependecies and run the notebook and a modelfile will be exported into your respective folder 
Then run the main.py file from terminal using streamlit run main.py and you will have a running website to upload images and predict the diseases