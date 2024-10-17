import streamlit as st
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Setting HuggingFace API key and repo ID
sec_key = "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" 
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"  

# Initialize the generative model from HuggingFace
llm_gen = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=512,
    temperature=0.7,
    token=sec_key
)

# Symptom Checker Template
symptom_checker_template = '''
Given the following symptoms: {symptoms}, suggest possible health conditions.
Conditions:
'''
prompt_symptom_checker = PromptTemplate(
    input_variables=['symptoms'],
    template=symptom_checker_template
)

# Treatment Recommendation Template
treatment_template = '''
For the health condition: {condition}, suggest possible treatments and lifestyle changes.
Treatments:
'''
prompt_treatment = PromptTemplate(
    input_variables=['condition'],
    template=treatment_template
)

# Dietary Advice Template
dietary_advice_template = '''
Provide dietary recommendations for someone who is {health_condition}.
Recommendations:
'''
prompt_dietary_advice = PromptTemplate(
    input_variables=['health_condition'],
    template=dietary_advice_template
)

# Fitness Plan Template
fitness_plan_template = '''
Create a fitness plan for an individual who is {age} years old, weighs {weight} kg, and has the following goals: {goals}.
Fitness Plan:
'''
prompt_fitness_plan = PromptTemplate(
    input_variables=['age', 'weight', 'goals'],
    template=fitness_plan_template
)

# Streamlit UI
st.title("HealthAI - Your Health Assistant")

st.sidebar.title("Choose a Health Task")
task = st.sidebar.selectbox(
    "Task",
    (
        "Symptom Checker",
        "Treatment Recommendation",
        "Dietary Advice",
        "Fitness Plan"
    ),
)

# Symptom Checker
if task == "Symptom Checker":
    st.header("Symptom Checker")
    symptoms = st.text_area("Enter symptoms (e.g., headache, fatigue):")
    if st.button("Check Symptoms"):
        if symptoms:
            symptom_chain = LLMChain(llm=llm_gen, prompt=prompt_symptom_checker)
            response = symptom_chain.run({"symptoms": symptoms})
            st.write("### Possible Health Conditions:")
            st.write(response)
        else:
            st.write("Please enter symptoms.")

# Treatment Recommendation
elif task == "Treatment Recommendation":
    st.header("Treatment Recommendation")
    condition = st.text_input("Enter health condition (e.g., diabetes, hypertension):")
    if st.button("Get Treatment Recommendations"):
        if condition:
            treatment_chain = LLMChain(llm=llm_gen, prompt=prompt_treatment)
            response = treatment_chain.run({"condition": condition})
            st.write("### Suggested Treatments and Lifestyle Changes:")
            st.write(response)
        else:
            st.write("Please enter a health condition.")

# Dietary Advice
elif task == "Dietary Advice":
    st.header("Dietary Advice")
    health_condition = st.text_input("Enter health condition (e.g., heart disease, obesity):")
    if st.button("Get Dietary Recommendations"):
        if health_condition:
            dietary_chain = LLMChain(llm=llm_gen, prompt=prompt_dietary_advice)
            response = dietary_chain.run({"health_condition": health_condition})
            st.write("### Dietary Recommendations:")
            st.write(response)
        else:
            st.write("Please enter a health condition.")

# Fitness Plan
elif task == "Fitness Plan":
    st.header("Fitness Plan")
    age = st.number_input("Enter your age:", min_value=1)
    weight = st.number_input("Enter your weight (kg):", min_value=1)
    goals = st.text_area("Enter your fitness goals (e.g., weight loss, muscle gain):")
    if st.button("Create Fitness Plan"):
        if goals:
            fitness_chain = LLMChain(llm=llm_gen, prompt=prompt_fitness_plan)
            response = fitness_chain.run({"age": age, "weight": weight, "goals": goals})
            st.write("### Fitness Plan:")
            st.write(response)
        else:
            st.write("Please enter your fitness goals.")

# Footer and Contact Info
st.sidebar.info("Developed by Rachit Ranjan.")
