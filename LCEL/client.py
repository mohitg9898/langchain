import requests
import streamlit as st


def get_groq_response(input_text):
    json_body = {
        "input": {
            "language": "French",
            "text": input_text
        }
    }
    try:
        response = requests.post("http://127.0.0.1:8000/chain/invoke", json=json_body)
        if response.status_code != 200:
            return {"error": f"Server error: {response.text}"}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

## Streamlit app
st.title("LLM Application Using LCEL")
input_text=st.text_input("Enter the text you want to convert to french")

if input_text:
    st.write(get_groq_response(input_text))