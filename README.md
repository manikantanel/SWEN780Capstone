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

Data will be stored in the Data_collection/ folder automatically.

Step 2: Train the Model

python train_model.py

This trains the LSTM model and saves it as action.h5. Accuracy/loss plots and a confusion matrix will be shown.

Step 3: Run Real-Time Detection (CLI) for testing

python run.py

This will open your webcam and show real-time detected words on screen.

Step 4: Run the Flask Web App

python app.py

Then go to:

http://localhost:5000

You’ll see a live webcam feed and real-time detected gestures displayed in text.

If you want to test setup, 

You can verify all imports are installed with:

python test_requirement.py

Author: Manikantan Eakiri Lakshmanan
Email:me2083@rit.edu

