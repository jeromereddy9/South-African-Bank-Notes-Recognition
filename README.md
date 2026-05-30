# South African Bank Notes Recognition

An image processing system for classifying South African banknotes (R10, R20, R50, R100, R200) invariant to rotation, scale, and side. Implements three models (SIFT-FLANN, ResNet-18, SimCLR) with ensemble voting. Includes preprocessing, segmentation, feature extraction, and a Streamlit GUI.

## How to Run the Application

### 1. Install Dependencies

Download or clone the repository, then open a terminal or command prompt at that location. If you downloaded it as a ZIP file, unzip it first. Then navigate to the root repository folder:

**If you downloaded and unzipped the repository:**
```bash
cd South-African-Bank-Notes-Recognition-main
```

**If you cloned the repository using Git:**
```bash
cd South-African-Bank-Notes-Recognition
```

Then install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Navigate to the Application Directory

Next, navigate into the inner folder containing the application script:

**If you downloaded and unzipped the repository:**
```bash
cd South-African-Bank-Notes-Recognition-main
```

**If you cloned the repository using Git:**
```bash
cd South-African-Bank-Notes-Recognition
```

You can confirm you are in the correct location by checking that `South_African_Bank_Notes_Recognition.py` is present in the current directory.

### 3. Launch the Application

Run the following command to start the application:
```bash
streamlit run South_African_Bank_Notes_Recognition.py
```

Streamlit will display a local URL in the terminal (typically `http://localhost:8501`) and may open a browser window automatically. If it doesn't, copy the URL and paste it into your browser.


