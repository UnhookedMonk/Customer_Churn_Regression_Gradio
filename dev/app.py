import gradio as gr
import pandas as pd
import numpy as np
import pickle

#Load the model and encoder and scaler
model = pickle.load(open("dev/model.pkl", "rb"))
encoder = pickle.load(open("dev/encoder.pkl", "rb"))
scaler = pickle.load(open("dev/scaler.pkl", "rb"))

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
                gr.components.Radio(['Male', 'Female'], label='Select the gender'),
                gr.components.Number(label="Is the customer a Seniorcitizen; No=0 and Yes=1"),
                gr.components.Radio(['Yes', 'No'], label='Does the customer have a Partner?'),
                gr.components.Dropdown(['No', 'Yes'], label='Does the customer have any Dependents? '),
                gr.components.Number(label='Length of tenure (no. of months with Telco)'),
                gr.components.Radio(['No', 'Yes'], label='Does the customer have PhoneService? '),
                gr.components.Radio(['No', 'Yes'], label='Does the customer have MultipleLines'),
                gr.components.Radio(['DSL', 'Fiber optic', 'No'], label='Does the customer have InternetService'),
                gr.components.Radio(['No', 'Yes'], label='Does the customer have OnlineSecurity?'),
                gr.components.Radio(['No', 'Yes'], label='Does the customer have OnlineBackup?'),
                gr.components.Radio(['No', 'Yes'], label='Does the customer DeviceProtection?'),
                gr.components.Radio(['No', 'Yes'], label='Does the customer have TechSupport?'),
                gr.components.Radio(['No', 'Yes'], label='Does the customer have StreamingTV?'),
                gr.components.Radio(['No', 'Yes'], label='Does the customer have StreamingMovies?'),
                gr.components.Dropdown(['Month-to-month', 'One year', 'Two year'], label='which Contract does the customer use?'),
                gr.components.Radio(['Yes', 'No'], label='Does the customer prefer PaperlessBilling?'),
                gr.components.Dropdown(['Electronic check', 'Mailed check', 'Bank transfer (automatic)',
                                        'Credit card (automatic)'], label='Which PaymentMethod does the customer prefer?'),
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
def input_df_creator(gender,seniorcitizen,partner,dependents, tenure, phoneservice,multiplelines,internetservice,
                     onlinesecurity,onlinebackup,deviceprotection,techsupport,streamingtv,streamingmovies,
                     contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges):
    input_data = pd.DataFrame({
        'gender': [gender],
        'seniorcitizen': [seniorcitizen],
        'partner': [partner],
        'dependents': [dependents],
        'tenure': [tenure],
        'phoneservice': [phoneservice],
        'multiplelines': [multiplelines],
        'internetservice': [internetservice],
        'onlinesecurity': [onlinesecurity],
        'onlinebackup': [onlinebackup],
        'deviceprotection': [deviceprotection],
        'techsupport': [techsupport],
        'streamingtv': [streamingtv],
        'streamingmovies': [streamingmovies],
        'contract': [contract],
        'paperlessbilling': [paperlessbilling],
        'paymentmethod': [paymentmethod],
        'monthlycharges': [monthlycharges],
        'totalcharges': [totalcharges]
    }) 
    return input_data

# Define the function to be called when the Gradio app is run
def predict_churn(gender,seniorCitizen,partner,dependents, tenure, phoneservice,multiplelines,internetservice,
                     onlinesecurity,onlinebackup,deviceProtection,techsupport,streamingtv,streamingmovies,
                     contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges):
    input_df = input_df_creator(gender,seniorCitizen,partner,dependents, tenure, phoneservice,multiplelines,internetservice,
                     onlinesecurity,onlinebackup,deviceProtection,techsupport,streamingtv,streamingmovies,
                     contract,paperlessbilling,paymentmethod,monthlycharges,totalcharges)
    
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