# South African Bank Notes Recognition

An image processing system for classifying South African banknotes (R10, R20, R50, R100, R200) invariant to rotation, scale, and side. Implements three models (SIFT-FLANN, ResNet-18, SimCLR) with ensemble voting. Includes preprocessing, segmentation, feature extraction, and a Streamlit GUI.

How to Run the Application

1) To run the application, first download or clone the project repository to your computer. Open a terminal or command prompt and navigate to the folder where the project was downloaded or extracted. Once you are in the repository's root directory, install the required Python dependencies by executing the following command:

                 pip install -r requirements.txt

2) After the installation is complete, navigate to the directory containing the application script. Because the repository contains a nested folder structure, you must first be in the project root directory and then execute the following command:

   If you downloaded the repository as a ZIP file, run:

         cd South-African-Bank-Notes-Recognition-main/South-African-Bank-Notes-Recognition-main
   
   If you cloned the repository using Git, run:

         cd South-African-Bank-Notes-Recognition/South-African-Bank-Notes-Recognition

You can verify that you are in the correct location by checking that the file South_African_Bank_Notes_Recognition.py is present in the current directory.

3) Once you have navigated to the correct directory, launch the Streamlit web application using the following command:

       streamlit run South_African_Bank_Notes_Recognition.py

After the application starts, Streamlit will display a local URL in the terminal, typically http://localhost:8501, and may automatically open a web browser window. If a browser window does not open automatically, copy and paste the displayed URL into your web browser. The graphical user interface will then be available, allowing you to upload banknote images and run the recognition pipeline.


