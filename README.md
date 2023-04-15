# CoastalIQ

Coastal IQ is a Web based application developed for Miami AI's Future Hackathon 2023 to utilize Artificial Intelligence(AI) to solve four project theme - AI for education, AI for environmental protection,AI for local government and AI for concession optimization. CoastIQ is focuses on the theme of  AI for environmental protection. Currently designed as a web-based application that utilizes Artificial Intelligence(AI) to allow users to capture images of beach and coastal plants, animals and pollutants; the applications shares specific details such as the name,origin, native habitat, harzard information and determine what actions users should take with the captured image. This application serve as an initial prototype to model some of the features of costal IQ capabilities, whether as a fully developed web application or mobile application for future development.

## How to set up and run the webapp

### Set up the Python environment

1. Download and install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create a Python environment -- name it coastaliq: `conda create --name coastaliq python=3.9`.
3. After the environment is installed, activate it `conda activate coastaliq`.
3. Install the required packages

```bash
conda install requests flask jupyter numpy scipy matplotlib pandas scikit-learn
conda install pytorch==1.13.1 torchvision==0.14.1 -c pytorch
pip install transformers
```

### Check out the code

Go to the workspace folder and check out the project `git checkout https://github.com/cgourdet/coastal-iq.git`.

### Run the web app

1. Activate the conda environment `conda activate coastaliq`.
2. Go to the project folder after the code is checked out. Then go to the `webapp` folder.
3. From there, launch the app, `flask run`.
4. The web app can be accessed from the browser `http://127.0.0.1:5000/bootstrap`.
5. To quick, ctrl+c.

## Authors

* **Eric WU* - *Initial work* - [eric-wu](https://github.com/eric-wu)
* **Claudia Gourdet* - *Initial work* - [cgourdet](https://github.com/cgourdet)
* **K. P.* - *Initial work*
* **Giancarlo Brea* - *Initial work* - [Giogio2448](https://github.com/Giogio2448)
* **Daniel Ruso* - *Initial work*
