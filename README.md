# Meminto - The AI based Meeting Minutes Tool
- AI based Meeting Minutes tool

## Goal of Meminto
- Control over information
  - runs mostly locally 

## How to use
Clone the Meminto git repository to your local device and follow the instructions below. 
### Setup environment variables
#### Hugging Face Access Token
In order to download a pretrained `pyannote.audio` model for speaker diarization from Hugging Face you will need to accept their terms and get an Hugging Face access token. To do so follow the first three steps of the `TL;DR` at https://huggingface.co/pyannote/speaker-diarization.<br> 
Before running Meminto store your access token in an environment variable called `HUGGING_FACE_ACCESS_TOKEN`.<br>
Windows PowerShell: `$Env:HUGGING_FACE_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"`<br>
Linux/MacOS: `export HUGGING_FACE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN`

#### OpenAI API key
In order to make the project as accessible as possible the default LLM used in this project is OpenAIs 'GPT-3.5-Turbo'. Nevertheless, feel free to adapt the corresponding curl command in `transscript_to_meeting_minutes.py` to any LLM available to you (see [below](#switch-to-a-different-llm)).<br> 
However, in order to use the default 'GPT-3.5-Turbo' LLM you will need an OpenAI API key. If you do not have one get it here: https://platform.openai.com/account/api-keys.<br>
Before running Meminto store your OpenAI key in the environment variable `OPENAI_API_KEY`.<br>
Windows Powershell: `$Env:OPENAI_API_KEY = "YOUR_KEY"`<br>
Linux/MacOS: `export OPENAI_API_KEY=YOUR_KEY`
### Install requirements
It is recommended to use a Python version >=3.10.<br>
<br>
Before running Meminto you will have to install the requirements from `requirements.txt`. I recommended to use some kind of virtual environment in which you can install the requirements for this project.<br> 
You could for example use `pipenv` (https://pypi.org/project/pipenv/):<br>
Install pipenv<br>
`pip install --user pipenv`<br>
Create an new environment<br>
`pipenv install`<br>
Actiavte the new environment<br>
`pipenv shell`<br>
Install the requirements for Meminto<br>
`pip install -r requirements.txt`

## Switch to a different LLM 

## To Improve

### Easier LLM switching
### Performance
- pyannote/speaker-diarization
https://huggingface.co/pyannote/speaker-diarization
"Real-time factor is around 2.5% using one Nvidia Tesla V100 SXM2 GPU (for the neural inference part) and one Intel Cascade Lake 6248 CPU (for the clustering part).
In other words, it takes approximately 1.5 minutes to process a one hour conversation."
