import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from io import BytesIO
import statsmodels.api as sm

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

st.title("Data Explorer")
st.write("Data explorer app let's you see muliple csv files in interative mode and let you see their statistics and visualizaton")

uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True, type ="csv")
for uploaded_file in uploaded_files:
    with st.container():
      st.write("Data Description of file : ",uploaded_file.name)
      dataframe = pd.read_csv(uploaded_file)
      if dataframe is not None:
        st.write(f"File: **{uploaded_file.name}**")
        st.write(f"Size: **{dataframe.memory_usage(deep=True).sum()} bytes**")
        st.write(f"Shape: **{dataframe.shape[0]} rows, {dataframe.shape[1]} columns**")
      
      if st.checkbox('Show Data Cleaning Options'):
        if st.button('Remove Duplicates'):
            dataframe = dataframe.drop_duplicates()
            st.write("Duplicates removed!")
            
        if st.button('Fill Missing Values with Mean'):
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                dataframe[col].fillna((dataframe[col].mean()), inplace=True)
            st.write("Missing values filled with mean!")
        if st.button('Remove Null values'):
            dataframe = dataframe.dropna()
            st.write("Null values removed!")
      
      st.write("Dataframe overall view")
      st.dataframe(dataframe)
      st.write("Data types present in dataframe")
      st.table(dataframe.dtypes)
      
      if st.checkbox('Show Data Transformation Options'):
        transformation_options = st.multiselect(
            'Select transformations to apply',
            ['Normalize Numeric Data', 'One-Hot Encode Categorical Data']
        )

        if 'Normalize Numeric Data' in transformation_options:
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
            dataframe[numeric_cols] = (dataframe[numeric_cols] - dataframe[numeric_cols].mean()) / dataframe[numeric_cols].std()
            st.write("Numeric data normalized.")

        if 'One-Hot Encode Categorical Data' in transformation_options:
            categorical_cols = dataframe.select_dtypes(include=['object']).columns
            dataframe = pd.get_dummies(dataframe, columns=categorical_cols)
            st.write("Categorical data one-hot encoded.")

      st.dataframe(dataframe)

      st.write("Statistical Operations on dataframe")
      st.table(dataframe.describe())
      numerical = dataframe.select_dtypes(include=["number","float64"])
      cateogircal = dataframe.select_dtypes(include=["object_","object"])
      st.write("Seeing the correlation between numerical dataframe columns")
      st.table(numerical.corr())
      
      if st.checkbox('Show Advanced Statistical Analysis'):
        st.write("Advanced Statistical Analysis")

        # Linear Regression Analysis
        if st.checkbox('Perform Linear Regression Analysis'):
            numeric_cols = dataframe.select_dtypes(include=[np.number]).columns
            st.write("Numeric Columns:", numeric_cols)

            # Selecting the predictor and response variables
            x_var = st.selectbox('Select the predictor variable', numeric_cols)
            y_var = st.selectbox('Select the response variable', numeric_cols)

            if st.button('Run Linear Regression'):
                X = sm.add_constant(dataframe[x_var]) # adding a constant
                Y = dataframe[y_var]

                model = sm.OLS(Y, X).fit()
                predictions = model.predict(X)

                st.write(model.summary())
                
      option_numerical = st.multiselect('Select which numerical columns to keep',list(numerical.columns))
      st.write('You selected:', option_numerical)
      option_categorical = st.multiselect('Select which categorical columns to keep',list(cateogircal.columns))
      st.write('You selected:', option_categorical)
      
      
      # For categorical data
      if len(option_categorical) > 0:
          fig = px.bar(cateogircal, x=option_categorical)
          st.plotly_chart(fig)

      # For numerical data
      if len(option_numerical) > 0:
          fig = px.line(numerical, x=option_numerical)
          st.plotly_chart(fig)
          
      df_xlsx = to_excel(dataframe)
      st.download_button(label='📥 Download Current Result',
                   data=df_xlsx ,
                   file_name= 'dataframe.xlsx')


with st.form("my_form"):
    st.write("Feedback")
    feedback = st.text_area("Share your feedback or request a feature")
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Thanks for your feedback!")

    