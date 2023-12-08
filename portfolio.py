import streamlit as st
import pickle
import numpy as np

# Load the trained models and their accuracies
def load_model(dataset_name, model_name):
    filename = f"{model_name.replace(' ', '_').lower()}_{dataset_name}_model.pkl"
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data['model'], data['accuracy']

def display_dataset_image(dataset_name):
    image_path = f"{dataset_name}_image.jpeg"  # Construct the image file path
    image = st.image(image_path, use_column_width=True)  # Display the image

# Main function to run the app
def main():
    st.title('Machine Learning Classifier Web App')
    
    # Select dataset
    dataset_name = st.sidebar.selectbox('Select Dataset', ['wine', 'iris'])
    display_dataset_image(dataset_name)

    
    # Select model
    model_name = st.sidebar.selectbox('Select Model', ['Logreg', 'SVM', 'DTree', 'RF', 'GBM', 'AdaBoost', 'kNN'])
    
    # Load the trained model and its accuracy based on user's selections
    model, model_accuracy = load_model(dataset_name, model_name)
    
    # Input form to get features from the user
    st.write(f'### Enter Features for {dataset_name} dataset')
    
    # Customize this input form based on the features of each dataset
    # For example, for the iris dataset:
    if dataset_name == 'iris':
        sepal_length = st.slider('Sepal Length', 4.0, 8.0, 5.4)
        sepal_width = st.slider('Sepal Width', 2.0, 4.5, 3.4)
        petal_length = st.slider('Petal Length', 1.0, 7.0, 4.7)
        petal_width = st.slider('Petal Width', 0.1, 2.5, 1.5)
        features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    if dataset_name == 'wine':
        fixed_acidity = st.slider('Fixed Acidity', 3.8, 15.9, 7.0)  
        volatile_acidity = st.slider('Volatile Acidity', 0.08, 1.58, 0.5)
        citric_acid = st.slider('Citric Acid', 0.0, 1.66, 0.25)
        residual_sugar = st.slider('Residual Sugar', 0.6, 65.8, 2.5)
        chlorides = st.slider('Chlorides', 0.009, 0.611, 0.05)
        free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', 1, 289, 30)
        total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', 6, 440, 100)
        density = st.slider('Density', 0.98711, 1.03898, 0.995)
        pH = st.slider('pH', 2.72, 4.01, 3.0)
        sulphates = st.slider('Sulphates', 0.22, 2.0, 0.5)
        alcohol = st.slider('Alcohol', 8.0, 14.9, 10.0)
        wine_type = st.selectbox('Type', ['red', 'white'])
        model_name = 'JohnBolorinos_WineFraud_SVM_model.plk'
    
        # Convert wine_type to a format suitable for the model (e.g., one-hot encoding)
        if wine_type == 'red':
            type_white = 0
        else:
            type_white = 1

        features = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, type_white]).reshape(1, -1)
    
        # Load the saved scaler and scale the input features
        with open('wine_scaler.plk', 'rb') as file:
            scaler = pickle.load(file)
            features = scaler.transform(features)


    if dataset_name == 'bank_retirement':
        age = st.slider('Age', 20, 100, 50)  # Adjust the range as needed
        savings_401k = st.slider('401K Savings ($)', 0, 1000000, 200000)  # Adjust the range and default value as needed
        features = np.array([age, savings_401k]).reshape(1, -1)

    # Adding input features for diabetes dataset
    if dataset_name == 'diabetes':
        pregnancies = st.slider('Pregnancies', 0, 17, 3)
        glucose = st.slider('Glucose', 0, 199, 117)
        blood_pressure = st.slider('Blood Pressure', 0, 122, 72)
        skin_thickness = st.slider('Skin Thickness', 0, 99, 23)
        insulin = st.slider('Insulin', 0, 846, 30)
        bmi = st.slider('BMI', 0.0, 67.1, 32.0)
        diabetes_pedigree = st.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
        age = st.slider('Age', 21, 81, 29)
    
        features = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
    
        # Load the saved scaler and scale the input features for diabetes (if you have a scaler for it)
        with open('diabetes_scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
            features = scaler.transform(features)


    # Predict button
    if st.button('Predict'):
        prediction = model.predict(features)
        # Check if the selected dataset is 'iris'
        if dataset_name == 'iris':
            iris_classes = ['Setosa', 'Versicolour', 'Virginica']
            flower_name = iris_classes[prediction[0]]
            st.write(f'### Prediction: {flower_name}')
        if dataset_name == 'wine':
            wine_classes = ['Fraud', 'Legit']
            wine_name = wine_classes[prediction[0]]
            st.write(f'### Prediction: {wine_name}')

    
        # Display the model's accuracy (assuming you've loaded or calculated the accuracy)
        st.write(f'### Model Accuracy: {model_accuracy:.2f}')



if __name__ == '__main__':
    main()
