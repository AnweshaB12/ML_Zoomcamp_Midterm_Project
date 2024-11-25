# ML_Zoomcamp_Midterm_Project
ML Zoomcamp Midterm Project on Weight Class Prediction

ML ZOOMCAMP MIDTERM PROJECT: WEIGHT CLASS PREDICTION

Problem Statement

Obesity is a critical global health issue that results from a combination of genetic, environmental, social, and psychological factors. Identifying individuals at risk of obesity is crucial to designing effective prevention strategies. This project focuses on predicting the weight class of an individual (underweight, normal, overweight, or obesity) based on features such as age, height, eating habits, and other social factors. By understanding which individuals are more susceptible to obesity, interventions can be targeted more effectively to mitigate this growing health crisis.

Role of Machine Learning

Machine learning is well-suited to solving this problem as it can uncover patterns and relationships within complex datasets that traditional statistical methods might overlook. By training models on labeled data, machine learning algorithms can predict weight class accurately and efficiently, even in the presence of multiple interacting factors. Furthermore, it allows for scalability and continuous improvement with new data, making it a valuable tool in combating obesity.

Overview of Classifiers Used

	1.	Decision Tree: This model splits the data into branches based on feature values, forming a tree-like structure to make predictions. It is intuitive and interpretable but can overfit if not properly regularized.
	2.	Random Forest Classifier: An ensemble of decision trees that combines predictions from multiple trees to improve accuracy and reduce overfitting. It is robust and performs well on multiclass problems.
	3.	Support Vector Machine (SVM): SVM finds the hyperplane that best separates classes in the feature space. It is effective for high-dimensional data but can be computationally expensive for large datasets.
	4.	K-Nearest Neighbors (KNN): This model classifies a sample based on the majority class of its nearest neighbors. It is simple and effective but can struggle with high-dimensional data and imbalanced datasets.

I used the above algorithms to train and cross-validate my data and then test the results in a Jupyter Notebook environment (Midterm_Project.ipynb) where I have performed exploratory data analysis to see which features have bigger impacts on the weight class (target).

Results

After testing and comparing the models, Random Forest achieved the highest accuracy , demonstrating its effectiveness in handling the complexity of the dataset and providing reliable predictions for weight class classification. This indicates its potential for real-world applications in identifying individuals at risk of obesity. So for deployment and running in a container using Flask, Gunicorn, and Docker, I chose the Random Forest Classifier model trained on my data.

FOR COMPILING THE PROJECT ON YOUR OWN MACHINE:

As stated before, this project predicts an individual’s weight class (underweight, normal, overweight, or obesity) based on their age, height, eating habits, and social factors using machine learning. It involves training a model, deploying it using Flask and Gunicorn, and running it in a Docker container.

Features

	•	Training a multiclass classification model using Random Forest Classifier.
	•	Deployment of the prediction model with Flask and Gunicorn.
	•	Containerized deployment using Docker for easy portability.

Dataset

The dataset (Obesity_Dataset.csv) was sourced from Kaggle and contains information on 1610 individuals. It is available in the project repository along with the processed files. There are only 2 numerical features (age, height) and the rest are categorical features which are initially represented with numbers. I converted these to the corresponding textual meanings. The dataset contains no missing values.

Setup Instructions:

1. Clone the Repository:

git clone https://github.com/yourusername/weight-class-prediction.git
cd weight-class-prediction

2. Install Dependencies in a Virtual Environment

i.	Create a new Conda environment:

conda create -n ml-zoomcamp python=3.10 -y

ii. Activate the environment:

conda activate ml-zoomcamp

iii. Install required Python libraries:

pip install -r requirements.txt

3. Train the Model

Run the training script to train the model and save it and the DictVectoriser as a .bin file:

python train.py

4. Test Locally Using Flask

i. Run flask app locally

python predict.py

ii. Test predictions using the predict-test.py script:

python predict-test.py

iii. Deploy Using Gunicorn

gunicorn --bind 0.0.0.0:8080 predict:app

5. Run with Docker

i. Build the image (defined in Dockerfile)

docker build -t weight-class-app .

ii. Run it:

docker run -it -p 8080:8080 weight_class_app




