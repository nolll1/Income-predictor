import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
model_path = r'C:\Users\Acer\Documents\incomeproject\ML_MODEL\logistic_regression_model.pkl'
model = joblib.load(model_path)

# Extract feature names from the model if they are available
try:
    expected_columns = model.feature_names_in_
except AttributeError:
    st.error("The model does not contain feature names. Please ensure the model is trained with feature names.")

def main():
    # Set the title of the web app
    st.title('INCOME Prediction')

    # Add a description
    st.write('Enter your information to predict your income.')

    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader('Your Information')

        # Add input fields for new features
        your_name = st.text_input('Full Name')
        age =st.slider("Your Age", 18,100)
        race = st.selectbox("Your Race", ['White', 'Black'])
        workclass = st.selectbox("Your workclass", ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov'])
        fnlwgt = st.number_input("Your Final Weight", min_value=0, max_value=1000000, value=0)
        education = st.selectbox("Education Level", ['10th', '11th', '12th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'Some-college','1st-4th','5th-6th'])
        education_num = st.slider("Your Education Number", 0, 16, 0)
        marital_status = st.selectbox("Your Marital Status", ['Divorced', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])
        occupation = st.selectbox("Your Occupation", [
            'Adm-clerical',
            'Craft-repair',
            'Exec-managerial',
            'Farming-fishing',
            'Handlers-cleaners',
            'Machine-op-inspct',
            'Other-service',
            'Prof-specialty',
            'Protective-serv',
            'Sales',
            'Tech-support',
            'Transport-moving'
            ])
        relationship = st.selectbox("Your Relationship", ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'])
        sex = st.selectbox("Your Sex", ['Female', 'Male'])
        capital_loss = st.number_input("Your Capital Loss", min_value=0, max_value=100000, value=0)
        hours_per_week = st.slider("Your Hours Per Week", 0, 100, 40)


    # Convert categorical inputs to numerical
    workclass = {'Federal-gov': 0, 'Local-gov': 1, 'Private': 2, 'Self-emp-inc': 3, 'Self-emp-not-inc': 4, 'State-gov': 5}.get(workclass, 0)
    education = {
        'education_10th': 0,
        'education_11th': 1,
        'education_12th': 2,
        'education_7th-8th': 3,
        'education_9th': 4,
        'education_Assoc-acdm': 5,
        'education_Assoc-voc': 6,
        'education_Bachelors': 7,
        'education_Doctorate': 8,
        'education_HS-grad': 9,
        'education_Masters': 10,
        'education_Prof-school': 11,
        'education_Some-college': 12,
        'education_1st-4th': 13,
        'education_5th-6th': 14,
    }.get(education, 0)
    marital_status = {
        'marital.status_Divorced': 0,
        'marital.status_Married-civ-spouse': 1,
        'marital.status_Married-spouse-absent': 2,
        'marital.status_Never-married': 3,
        'marital.status_Separated': 4,
        'marital.status_Widowed': 5
    }.get(marital_status, 0)
    occupation = {
        'occupation_Adm-clerical': 0,
        'occupation_Craft-repair': 1,
        'occupation_Exec-managerial': 2,
        'occupation_Farming-fishing': 3,
        'occupation_Handlers-cleaners': 4,
        'occupation_Machine-op-inspct': 5,
        'occupation_Other-service': 6,
        'occupation_Prof-specialty': 7,
        'occupation_Protective-serv': 8,
        'occupation_Sales': 9,
        'occupation_Tech-support': 10,
        'occupation_Transport-moving': 11
    }.get(occupation, 0)
    relationship = {
    'relationship_Husband': 0,
    'relationship_Not-in-family': 1,
    'relationship_Other-relative': 2,
    'relationship_Own-child': 3,
    'relationship_Unmarried': 4,
    'relationship_Wife': 5
    }.get(relationship, 0)
    sex = 1 if sex =='Female' else 0


    # Prepare input data as a DataFrame
    input_data = pd.DataFrame({
            'workclass_Federal-gov': [1 if workclass == 0 else 0],
            'workclass_Local-gov': [1 if workclass == 1 else 0],
            'workclass_Private': [1 if workclass == 2 else 0],
            'workclass_Self-emp-inc': [1 if workclass == 3 else 0],
            'workclass_Self-emp-not-inc': [1 if workclass == 4 else 0],
            'workclass_State-gov': [1 if workclass == 5 else 0],
            'education_10th': [1 if education == 0 else 0],
            'education_11th': [1 if education == 1 else 0],
            'education_12th': [1 if education == 2 else 0],
            'education_7th-8th': [1 if education == 3 else 0],
            'education_9th': [1 if education == 4 else 0],
            'education_Assoc-acdm': [1 if education == 5 else 0],
            'education_Assoc-voc': [1 if education == 6 else 0],
            'education_Bachelors': [1 if education == 7 else 0],
            'education_Doctorate': [1 if education == 8 else 0],
            'education_HS-grad': [1 if education == 9 else 0],
            'education_Masters': [1 if education == 10 else 0],
            'education_Prof-school': [1 if education == 11 else 0],
            'education_Some-college': [1 if education == 12 else 0],
            'education_1st-4th': [1 if education == 13 else 0],
            'education_5th-6th': [1 if education == 14 else 0],
            'marital.status_Divorced': [1 if marital_status == 0 else 0],
            'marital.status_Married-civ-spouse': [1 if marital_status == 1 else 0],
            'marital.status_Married-spouse-absent': [1 if marital_status == 2 else 0],
            'marital.status_Never-married': [1 if marital_status == 3 else 0],
            'marital.status_Separated': [1 if marital_status == 4 else 0],
            'marital.status_Widowed': [1 if marital_status == 5 else 0],
            'occupation_Adm-clerical': [1 if occupation == 0 else 0],
            'occupation_Craft-repair': [1 if occupation == 1 else 0],
            'occupation_Exec-managerial': [1 if occupation == 2 else 0],
            'occupation_Farming-fishing': [1 if occupation == 3 else 0],
            'occupation_Handlers-cleaners': [1 if occupation == 4 else 0],
            'occupation_Machine-op-inspct': [1 if occupation == 5 else 0],
            'occupation_Other-service': [1 if occupation == 6 else 0],
            'occupation_Prof-specialty': [1 if occupation == 7 else 0],
            'occupation_Protective-serv': [1 if occupation == 8 else 0],
            'occupation_Sales': [1 if occupation == 9 else 0],
            'occupation_Tech-support': [1 if occupation == 10 else 0],
            'occupation_Transport-moving': [1 if occupation == 11 else 0],
            'relationship_Husband': [1 if relationship == 0 else 0],
            'relationship_Not-in-family': [1 if relationship == 1 else 0],
            'relationship_Other-relative': [1 if relationship == 2 else 0],
            'relationship_Own-child': [1 if relationship == 3 else 0],
            'relationship_Unmarried': [1 if relationship == 4 else 0],
            'relationship_Wife': [1 if relationship == 5 else 0],
            'sex_Female': [sex],
            'sex_Male': [1 - sex],
            'age': [age],
            'fnlwgt': [fnlwgt],
            'education.num': [education_num],
            'capital.loss': [capital_loss],
            'hours.per.week': [hours_per_week]
        })

    # Ensure columns are in the same order as during model training
    input_data = input_data[expected_columns]

    # Prediction and results section
    with col2:
        st.subheader('Prediction')
        if st.button('Predict'):
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1]
        
            st.write(f'Prediction for {your_name}: {">50K" if prediction[0] == 1 else "<=50K"}')
            st.write(f'Probability of earning more: {probability:.2f}')

            # Plotting
            fig, axes = plt.subplots(3, 1, figsize=(8, 16))

            # Plot Income probability
            sns.barplot(x=['<=50K', '>50K'], y=[1 - probability, probability], ax=axes[0], palette=['red', 'green'])
            axes[0].set_title('Income Probability')
            axes[0].set_ylabel('Probability')

            # Plot Income distribution
            sns.histplot(input_data['hours.per.week'], kde=True, ax=axes[1])
            axes[1].set_title('Income Distribution')

            # Plot Income pie chart
            axes[2].pie([1 - probability, probability], labels=['<=50K', '>50K'], autopct='%1.1f%%', colors=['red', 'green'])
            axes[2].set_title('Income Pie Chart')

            # Display the plots
            st.pyplot(fig)

            # Provide recommendations
            if prediction[0] == 1:
                st.success(f"{your_name} is likely to earn more than 50K. Keep up the good work!")
            else:
                st.error(f"{your_name} is likely to earn less than 50K. Consider improving your skills to achieve a better job.")

if __name__ == '__main__':
    main()