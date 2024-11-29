
import ast
import tempfile
import pymupdf4llm
import pymupdf
import datetime
from autogen import ConversableAgent
from Report_generator import *
from main_model import *
from overlay_cam import *


# gemma={
#     "config_list" : [
#     {
#         # Let's choose the Meta's Llama 3.1 model (model names must match Ollama exactly)
#         "model": "meditron",
#         # We specify the API Type as 'ollama' so it uses the Ollama client class
#         "api_type": "ollama",
#         "stream": False,
#         "client_host": "http://127.0.0.1:11434",
#     }
# ]
# }


##############################################################
# patient symptoms
# use directly will think layer if i want to extract symptoms or not

################################################################
#medical imaging


# xray_agent = ConversableAgent(
#     "blood_test",
#     system_message="You are a doctor, your work is to extract the results of the complete blood count test along with reference value, and also mention if its high or low, just print results nothing else",
#     llm_config=gemma,
#     code_execution_config=False,  # Turn off code execution, by default it is off.
#     function_map=None,  # No registered functions, by default it is None.
#     human_input_mode="NEVER",  # Never ask for human input.
# )

# cbc_report = cbc_agent.generate_reply(messages=[{"content": md_text, "role": "user"}])
# print(cbc_report)


##########################################################
# main model

# doctor_agent = ConversableAgent(
#     "Doctor",
#     system_message="You are a doctor, I have provided you patient history, his blood test result (if any) and x ray result (if any), using this information , identify which disease the patient might be suffering from, give response in following dictionary format {response: your response , logs: here you will give explaination of why you gave this result } ",
#     llm_config=gemma,
#     code_execution_config=False,  # Turn off code execution, by default it is off.
#     function_map=None,  # No registered functions, by default it is None.
#     human_input_mode="NEVER",  # Never ask for human input.
# )

# doctor_report = doctor_agent.generate_reply(messages=[{"content": md_text, "role": "user"}])
# print(doctor_report)


########################################################
# Blood test

# cbc_agent = ConversableAgent(
#     "blood_test",
#     system_message="You are a doctor, your work is to extract the results of the complete blood count test along with reference value, and also mention if its high or low, just print results nothing else",
#     llm_config=gemma,
#     code_execution_config=False,  # Turn off code execution, by default it is off.
#     function_map=None,  # No registered functions, by default it is None.
#     human_input_mode="NEVER",  # Never ask for human input.
# )

# cbc_report = cbc_agent.generate_reply(messages=[{"content": md_text, "role": "user"}])
# print(cbc_report)


import streamlit as st

# Initialize patient history in session state
if "patient_history" not in st.session_state:
    st.session_state.patient_history = {}

# Sidebar: Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Form Page","Document Query", "History"])

# Main content
if page == "Form Page":
    # Form Page
    st.title("🩺 Patient Diagnosis Form")
    st.write("Please fill in the following details to help us better assess your condition.")

    # Patient Details Section
    st.header("1. Patient Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Name:", placeholder="Enter your full name")
    with col2:
        gender = st.selectbox("Gender:", ["Male", "Female", "Other", "Prefer not to say"])
    with col3:
        age = st.number_input("Age:", min_value=0, max_value=120, step=1, format="%d")

    # Symptoms Section
    st.header("2. Patient Symptoms")
    symptoms = st.text_area(
        "Describe your symptoms in detail:",
        placeholder="E.g., fever, cough, chest pain, etc."
    )

    # Upload Blood Test Report Section
    st.header("3. Upload Blood Test Report")
    blood_test = st.file_uploader(
        "Upload your blood test report (optional):",
        type=["pdf"],
        help="Supported formats: PDF"
    )

    # Upload Imaging Section
    st.header("4. Upload Imaging Files")
    imaging_files = st.file_uploader(
        "Upload your imaging files (optional):",
        type=["pdf", "jpg", "png", "jpeg", "dcm"],
        help="Supported formats: PDF, JPG, PNG, JPEG"
    )

    model_category = st.selectbox(
        "Select inference model:",
        ["Gemma:2b", "Medllama3:7b", "Meditron:7b", "BioMistral:7b", "GPT4"]
    )

    # Submission Button
    if st.button("Submit"):
        if not name:
            st.error("Please provide your name.")
        elif not age:
            st.error("Please provide your age.")
        elif not symptoms:
            st.error("Please describe your symptoms.")
        else:
            st.success("Form submitted successfully!")
            st.write("### Summary of Your Submission:")
            st.write(f"**Patient:** {name}, {gender}, {age} ")
            st.write(f"**Symptoms Described:** {symptoms}")
            st.write(f"**Model:** {model_category}")

            st.session_state.patient_history[name] = {
                "name": name,
                "age": age,
                "gender": gender,
                "symptoms": symptoms,
                "blood_test": "Uploaded" if blood_test else "Not provided",
                "imaging_files": "Uploaded" if imaging_files else "Not provided",
                "model": model_category,
            }


            if blood_test:
                st.write("**Blood Test Report:** Uploaded")
                data = blood_test.getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                  temp_file.write(data)
                  cbc_report = pymupdf4llm.to_markdown(temp_file.name)
                  # cbc_report = cbc_agent.generate_reply(messages=[{"content": cbc_report, "role": "user"}])

            else:
                st.write("**Blood Test Report:** Not provided")
                cbc_report="none"

            if imaging_files:
                st.write(f"**Imaging Files:** Uploaded:")

                data_xray = imaging_files.getvalue()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file_xray:
                    temp_file_xray.write(data_xray)
                    xray=xray_output()

            else:
                st.write("**Imaging Files:** Not provided")
                xray="none"

            if model_category == "Gemma:2b":
                model_use="lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf"
            elif model_category == "Medllama3:7b":
                model_use="bartowski/JSL-MedLlama-3-8B-v2.0-GGUF/JSL-MedLlama-3-8B-v2.0-Q4_K_S.gguf"

            elif model_category == "BioMistral:7b":
                model_use="MaziyarPanahi/BioMistral-7B-GGUF/BioMistral-7B.Q3_K_S.gguf"
            gemma = {
                "config_list": [
                    {
                        "model": f"{model_use}",
                        "base_url": "http://localhost:1234/v1",
                        "api_key": "not-needed",
                    },
                ],
            }

            doctor_agent = ConversableAgent(
                "Doctor",
                system_message="You are a doctor, and your role is to diagnose a patient based on the symptoms provided. I will describe the symptoms, and you need to tell me the potential disease the patient might be suffering from, with a clear and detailed answer and also tell why you think that the patient might be suffering from that. Do not provide any other information or dialogue besides the diagnosis. Provide the diagnosis in full immediately after receiving the symptoms.",
                llm_config=gemma,
                code_execution_config=False,  # Turn off code execution, by default it is off.
                function_map=None,  # No registered functions, by default it is None.
                human_input_mode="NEVER",  # Never ask for human input.
            )


            # doctor_report = main_inference(model_category,symptoms,cbc_report,xray)
            doctor_report = doctor_agent.generate_reply(messages=[{"role": "user","content": f"symptoms are {symptoms}, blood test report is {cbc_report} , and xray result {xray} . what disease he is suffering from ? tell in detail and also tell insights from blood test data. In the end give reference from relevant blood test data along with values"}])
            # doctor_report=doctor_report
            st.write("### Diagnosis Report:")
            st.write(doctor_report)


            report_gen(name,age,gender,symptoms,doctor_report,str(datetime.datetime.now().date()),xray)

            if imaging_files:
                file_path = f"Records/{name}_full.pdf"
            else:
                file_path = f"Records/{name}1.pdf"
            # file_path = f"{name}.pdf"

            with open(file_path, "rb") as file:
              file_data = file.read()

            # Add a download button
            st.download_button(
                label="Download AI Diagnosis PDF",
                data=file_data,
                file_name=f"{name}_diagnosis.pdf",
                mime="application/pdf"
            )


elif page == "Document Query":
    # Dynamic Page
    st.title("📑 Document Query")
    st.write("Query over you documents here")

    rag = st.file_uploader(
        "Upload your document here",
        type=["pdf", "jpg", "png", "jpeg"],
        help="Supported formats: PDF")

    query = st.text_area(
        "Enter your query:",
        placeholder="E.g., what is the diagnosis test for"
    )


    if st.button("Submit"):
      if not rag:
        st.error("Please provide a document")
      else:
        st.success("submitted successfully! Querying... ")
        st.write("### Summary of Your Submission:")







elif page == "History":
    # Dynamic Page
    st.title("📊 Patient Statistics & Insights")
    st.write("This page dynamically displays aggregated patient data.")

    if st.session_state.patient_history:
        st.write("### Patient History Overview")
        for name, data in st.session_state.patient_history.items():
            st.write(f"**Name:** {name}")
            st.write(f"**Age:** {data['age']} | **Gender:** {data['gender']}")
            st.write(f"**Symptoms:** {data['symptoms']}")

            file_path = f"Records/{name}_full.pdf"
            with open(file_path, "rb") as file:
              file_data = file.read()

            # Add a download button
            st.download_button(
                label="Download AI Diagnosis PDF",
                data=file_data,
                file_name=f"{name}_diagnosis.pdf",
                mime="application/pdf"
            )

            st.write("---")



    else:
        st.warning("No patient data available. Please add patients from the Form Page.")

# Footer
st.markdown("---")

