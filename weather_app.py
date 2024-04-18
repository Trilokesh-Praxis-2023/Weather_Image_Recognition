import streamlit as st
import subprocess

def reboot_streamlit_app():
    # Execute a command to restart the Streamlit server
    subprocess.Popen(["streamlit", "run", "weather_app.py", "--browser.serverAddress=0.0.0.0", "--server.runOnSave=True"])
    # Exit the current script
    exit()

# Streamlit app
def main():
    st.title("Reboot Streamlit App")
    st.write("Click the button below to reboot the Streamlit app.")
    
    if st.button("Reboot"):
        reboot_streamlit_app()

if __name__ == "__main__":
    main()
