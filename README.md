# SWEN780Capstone

Installation & Setup

1. Clone the Repository

git clone https://github.com/manikantanel/SWEN780Capstone.git
cd asl-gesture-dictionary

2. Create & Activate a Virtual Environment

python3.8 -m venv asl_env
source asl_env/bin/activate  # On Windows: asl_env\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

How to work on this project

Step 1: Collect Gesture Data
Run the script and follow prompts to collect 30 video sequences for each gesture:

python collected_data.py

![image](https://github.com/user-attachments/assets/850de6b9-d18f-4bf9-b57a-ce8f94bb0eae)

Data will be stored in the Data_collection/ folder automatically.

Step 2: Train the Model

python train_model.py

This trains the LSTM model and saves it as action.h5. Accuracy/loss plots and a confusion matrix will be shown.

Step 3: Run Real-Time Detection (CLI) for testing

python run.py

![image](https://github.com/user-attachments/assets/f88e73c6-d3ed-4059-a5c6-a8b026dc4632)


This will open your webcam and show real-time detected words on screen.

Step 4: Run the Flask Web App

python app.py

Then go to:

http://localhost:5000

![image](https://github.com/user-attachments/assets/ddf4d57f-cf16-464a-8b93-32decd7820f0)

![image](https://github.com/user-attachments/assets/f413aa2a-c3e1-494f-8e0f-522083c1ff64)


You’ll see a live webcam feed and real-time detected gestures displayed in text.

If you want to test setup, 

You can verify all imports are installed with:

python test_requirement.py

Author: Manikantan Eakiri Lakshmanan
Email:me2083@rit.edu

