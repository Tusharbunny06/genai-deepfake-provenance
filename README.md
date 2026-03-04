\# GenAI Deepfake Detection with Provenance Verification



\## Overview



This project presents a \*\*Generative AI–based deepfake detection and provenance verification system\*\*.

It combines \*\*synthetic media generation, adversarial training, and detection models\*\* to identify whether a video is real or AI-generated and trace its origin.



The system also includes a \*\*Streamlit web interface\*\* that allows users to:



\* Generate AI videos using prompts

\* Upload videos for deepfake detection

\* Verify media provenance



\---



\## Features



\* Deepfake detection using a trained CNN model

\* AI video generation from text prompts

\* Adversarial example generation

\* Media provenance verification using hashing

\* Interactive Streamlit interface

\* Modular Python architecture



\---



\## Project Architecture



```

User Input

 │

 ├── Prompt → AI Video Generator
  │

 └── Upload Video

          │

        ▼

 Frame Extraction

        │

       ▼

  Deepfake Detection Model

         │

       ▼

  Provenance Verification

         │

        ▼

 Result Display (Streamlit UI)

```



\---



\## Project Structure



```

genai-deepfake-provenance

│

├── app.py                 # Streamlit web interface

├── video\_generator.py     # AI video generation

├── train\_detector.py      # Model training

├── evaluate.py            # Model evaluation

├── dataset.py             # Dataset loader

├── split\_data.py          # Train/val/test splitting

├── provenance.py          # Media provenance verification

├── generate\_adversarial.py

├── generate\_fakes.py

│

├── utils/                 # Helper utilities

├── notebooks/             # Experiment notebooks

│

├── requirements.txt

└── README.md

```



\---



\## Installation



Clone the repository:



```bash

git clone https://github.com/Tusharbunny06/genai-deepfake-provenance.git

cd genai-deepfake-provenance

```



Create a virtual environment:



```bash

python -m venv venv

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## Running the Application



Start the Streamlit interface:



```bash

streamlit run app.py

```



The application will open in your browser:



```

http://localhost:8501

```



\---



\## Usage



\### Generate AI Video



1\. Select \*\*Generate AI Video\*\*

2\. Enter a text prompt

3\. The system generates frames and composes them into a video



\### Detect Deepfake



1\. Select \*\*Upload Video\*\*

2\. Upload a video file

3\. The model predicts whether it is \*\*real or fake\*\*



\### Provenance Verification



The system computes \*\*hash-based provenance signatures\*\* to verify media authenticity.



\---



\## Technologies Used



\* Python

\* PyTorch

\* Streamlit

\* OpenCV

\* Diffusers

\* Transformers

\* HuggingFace



\---



\## Future Improvements



\* Real-time deepfake detection

\* Blockchain-based provenance tracking

\* Transformer-based detection models

\* Larger deepfake datasets



\---





