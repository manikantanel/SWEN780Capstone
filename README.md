# ASL to Text Dictionary Using Gesture Detection

Installation & Setup

1. Clone the Repository

        git clone https://github.com/manikantanel/SWEN780Capstone.git
2. Install Python 3.8.10 (if not already installed)
        This project requires Python 3.8.10. Follow these steps if it's not already installed:

        Go to the official Python release page:
        🔗 https://www.python.org/downloads/release/python-3810/

Scroll down and download: Windows x86-64 executable installer

Run the installer and make sure to:

* Check “Add Python 3.8 to PATH”

* Select “Customize Installation”

* Ensure “pip” and “venv” are checked

* Finish installation

Verify in terminal:

        python  --version

3. Create & Activate a Virtual Environment

        python -m venv asl_env
For Git Bash:

        source asl_env/Scripts/activate
For Windows Command Prompt:

        asl_env\Scripts\activate

4. Install Dependencies

        pip install -r requirements.txt

If you want to test setup, 

You can verify all imports are installed with:

        python test_requirement.py

How to work on this project

Step 1: Collect Gesture Data

        python collected_data.py
If you installed Python 3.8 manually and need to run it directly, use:

        "C:/Users/YourName/AppData/Local/Programs/Python/Python38/python.exe" "c:/Users/YourName/SWEN780Capstone/collected_data.py"
        
This collects 30 video sequences per gesture
Data will be stored in the Data_collection/ folder automatically.

![image](https://github.com/user-attachments/assets/850de6b9-d18f-4bf9-b57a-ce8f94bb0eae)


Step 2: Train the Model

        python train_model.py
Manual path version:

        "C:/Users/YourName/AppData/Local/Programs/Python/Python38/python.exe" "c:/Users/YourName/SWEN780Capstone/train_model.py"

This trains the LSTM model and saves it as action.h5. Accuracy/loss plots and a confusion matrix will be shown.

Step 3: Run Real-Time Detection (CLI) for testing

        python run.py
Manual path version:

        "C:/Users/YourName/AppData/Local/Programs/Python/Python38/python.exe" "c:/Users/YourName/SWEN780Capstone/run.py"

![image](https://github.com/user-attachments/assets/f88e73c6-d3ed-4059-a5c6-a8b026dc4632)


This will open your webcam and show real-time detected words on screen.

Step 4: Run the Flask Web App

        python app.py

Manual path version:

        "C:/Users/YourName/AppData/Local/Programs/Python/Python38/python.exe" "c:/Users/YourName/SWEN780Capstone/app.py"
Then go to:

        http://localhost:5000

You’ll see a live webcam feed and real-time detected gestures displayed in text.

![image](https://github.com/user-attachments/assets/ddf4d57f-cf16-464a-8b93-32decd7820f0)

![image](https://github.com/user-attachments/assets/f413aa2a-c3e1-494f-8e0f-522083c1ff64)



Author: Manikantan Eakiri Lakshmanan
Email:me2083@rit.edu

