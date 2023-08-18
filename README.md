# Meminto - The AI based Meeting Minutes Tool

Meminto is an AI based tool to create meeting minutes. Just hand it an '.wav' audio file of a recorded meeting and it will automatically generate your meeting minutes.<br>
- In a first step it will use speaker-diarization of pyannote.audio in order to differentiate between the different speakers.<br>
- It will then use whisper in order to generate an transcript of the meeting.<br>
- Finally, it will use an LLM (current defaul GPT-3.5-Turbo) to generate the meeting minutes. 

While there are a lot of commercially available tools to generate meeting notes, Meminto was intended to be an open source tool that gives the user control over its data.<br>
Therefore, the diarization and transcription are excuted on your local device. Note however, that for the final creation of the meeting minutes Meminto uses OpenAIs GPT-3.5-Turbo as default LLM. Thus, in order to ensure that your data are not leaked you should adapt the corresponding function  `transscript_to_meeting_minutes` in `transscript_to_meeting_minutes.py` to use the LLM of your choice that you trust with your data. 

## How to setup and run Meminto

### TL;DR
```shell
#1 clone Meminto repository
git clone https://github.com/FlorianSchepers/Meminto.git
cd Meminto
#2 Install requirements
pip install --user pipenv #if not alread installed
pipenv install
pipenv shell
pip install -r requirements.txt
#3 Set environment variables
export HUGGING_FACE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN #see https://huggingface.co/pyannote/speaker-diarization
export OPENAI_API_KEY=YOUR_KEY #see https://platform.openai.com/account/api-keys
#4 run Meminto
python -m main -f <file-path> #replace '<file-path>' with path to audio file 
```

### Detailed description 

#### Meminto repository
Clone the [Meminto](#https://github.com/FlorianSchepers/Meminto.git) repository by running and move the the top level folder<br>
```shell
git clone https://github.com/FlorianSchepers/Meminto.git
cd Meminto
```
#### Install requirements
I recommended to use a Python version >=3.10.<br>
Next, install the requirements from `requirements.txt`. I recommended to use some kind of virtual environment. You could for example use `pipenv` (https://pypi.org/project/pipenv/):<br>
```shell
pip install --user pipenv #Install pipenv
pipenv install #Create an new environment
pipenv shell #Activate the new environment
pip install -r requirements.txt #Install the requirements
```
#### Setup environment variables

##### Hugging Face Access Token

In order to download a pretrained `pyannote.audio` model for speaker diarization from Hugging Face you will need to accept their terms and get an Hugging Face access token. To do so follow the first three steps of the `TL;DR` at https://huggingface.co/pyannote/speaker-diarization.<br>
Before running Meminto store your access token in an environment variable called `HUGGING_FACE_ACCESS_TOKEN`.<br>
```Shell
export HUGGING_FACE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN #Linux/MacOS
```
```Powershell
$Env:HUGGING_FACE_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN" #Windows PowerShell
```


##### OpenAI API key

In order to make the project as accessible as possible the default LLM used in this project is OpenAIs 'GPT-3.5-Turbo'. Nevertheless, feel free to adapt the corresponding curl command in `transscript_to_meeting_minutes.py` to any LLM available to you.<br>
However, in order to use the default 'GPT-3.5-Turbo' LLM you will need an OpenAI API key. If you do not have one get it here: https://platform.openai.com/account/api-keys.<br>
Before running Meminto store your OpenAI key in the environment variable `OPENAI_API_KEY`.<br>
```Shell
export OPENAI_API_KEY=YOUR_KEY #Linux/MacOS
```
```Powershell
$Env:OPENAI_API_KEY = "YOUR_KEY" #Windows PowerShell
```

#### How to run Meminto
From the top level folder of Meminto run:
```shell
python -m main -f <file-path>
```
Where `<file-path>` corresponds to the path of the audio file for which you want to create the meeting minutes. 
