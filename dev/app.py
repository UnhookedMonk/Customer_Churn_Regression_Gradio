import gradio as gr
import pandas as pd
import numpy as np
import pickle

#Load the model and encoder and scaler
#model = pickle.load(open("model.pkl", "rb"))
#encoder = pickle.load(open("encoder.pkl", "rb"))
#scaler = pickle.load(open("scaler.pkl", "rb"))

#Load the data
data = pd.read_csv('assets/training_dataset.csv')

# Define the input and output interfaces for the Gradio app
input_interface=[]
with gr.Blocks(css=".gradio-container {background-color: powderblue}") as app:
#    img = gr.Image("telecom churn.png").style(height='13')

    Title=gr.Label('Customer Churn Prediction App')

    with gr.Row():
        Title
#    with gr.Row():
#        img

#with gr.Blocks() as app:
#    with gr.Blocks(css=".gradio-interface-container {background-color: powderblue}"):
        #with gr.Row():
        #    gr.Label('Customer Churn Prediction Model')
    with gr.Row():
        gr.Markdown("This app predicts whether a customer will leave your company or not. Enter the details of the customer below to see the result")

    #with gr.Row():
        #gr.Label('This app predicts whether a customer will leave your company or not. Enter the details of the customer below to see the result')


    with gr.Row():
        with gr.Column(scale=3, min_width=600):

            input_interface = [
                gr.components.Radio(['male', 'female'], label='Select your gender'),
                gr.components.Number(label="Are you a Seniorcitizen; No=0 and Yes=1"),
                gr.components.Radio(['Yes', 'No'], label='Do you have Partner'),
                gr.components.Dropdown(['No', 'Yes'], label='Do you have any Dependents? '),
                gr.components.Number(label='Lenght of tenure (no. of months with Telco)'),
                gr.components.Radio(['No', 'Yes'], label='Do you have PhoneService? '),
                gr.components.Radio(['No', 'Yes'], label='Do you have MultipleLines'),
                gr.components.Radio(['DSL', 'Fiber optic', 'No'], label='Do you have InternetService'),
                gr.components.Radio(['No', 'Yes'], label='Do you have OnlineSecurity?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have OnlineBackup?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have DeviceProtection?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have TechSupport?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have StreamingTV?'),
                gr.components.Radio(['No', 'Yes'], label='Do you have StreamingMovies?'),
                gr.components.Dropdown(['Month-to-month', 'One year', 'Two year'], label='which Contract do you use?'),
                gr.components.Radio(['Yes', 'No'], label='Do you prefer PaperlessBilling?'),
                gr.components.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                        'Credit card (automatic)'], label='Which PaymentMethod do you prefer?'),
                gr.components.Number(label="Enter monthly charges"),
                gr.components.Number(label="Enter total charges")
            ]

    with gr.Row():
        submit_btn = gr.Button('Submit')

        predict_btn = gr.Button('Predict')

output_components = [
   gr.Label(label="Churn Prediction"),
]

# Convert the input values to a pandas DataFrame with the appropriate column names
def input_df_creator(gender,SeniorCitizen,Partner,Dependents, tenure, PhoneService,MultipleLines,InternetService,
                     OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
                     Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]
    }) 
    return input_data

# Define the function to be called when the Gradio app is run
def predict_churn(gender,SeniorCitizen,Partner,Dependents, tenure, PhoneService,MultipleLines,InternetService,
                     OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
                     Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges):
    input_df = input_df_creator(gender,SeniorCitizen,Partner,Dependents, tenure, PhoneService,MultipleLines,InternetService,
                     OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,
                     Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges)
    
    # Encode categorical variables
    cat_cols = data.select_dtypes(include=['object']).columns
    cat_encoded = encoder.transform(input_df[cat_cols])

    # Scale numerical variables
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    num_scaled = scaler.transform(input_df[num_cols])

    # joining encoded and scaled columns back together
    processed_df = pd.concat([num_scaled, cat_encoded], axis=1)

    # Make prediction
    prediction = model.predict(processed_df)
    return "Churn" if prediction[0] == 1 else "No Churn"

# Launch the Gradio app
iface = gr.Interface(predict_churn, inputs=input_interface, outputs=output_components)
iface.launch(inbrowser= True, show_error= True)