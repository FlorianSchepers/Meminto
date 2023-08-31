# Meminto - The AI based Meeting Minutes Tool

Meminto is an AI based tool to create meeting minutes. Just hand it a '.wav' audio file of a recorded meeting and it will automatically generate your meeting minutes.<br>
- In a first step it will use speaker-diarization of pyannote.audio in order to differentiate between the different speakers.<br>
- It will then use whisper in order to generate an transcript of the meeting.<br>
- Finally, it will use an LLM to generate the meeting minutes. 

While there are a lot of commercially available tools to generate meeting notes, Meminto was intended to be an open source tool that gives the users control over their data.<br>
Therefore, the diarization and transcription are executed on your local device. Note however, that for the final creation of the meeting minutes Meminto uses OpenAIs GPT-3.5-Turbo as default LLM. Thus, in order to ensure that your data are not leaked you should adapt the corresponding function  `transscript_to_meeting_minutes` in `transscript_to_meeting_minutes.py` to use the LLM of your choice that you trust with your data. 

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

#Option 1: In case you want to use an arbitrary LLM 
export HUGGING_FACE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN #see TL;DR of https://huggingface.co/pyannote/speaker-diarization
export LLM_URL=YOUR_LLM_URL #e.g. "https://api.openai.com/v1/chat/completions" for openAI
export LLM_MODEL=YOUR_LLM_MODEL #e.g. "gpt-3.5-turbo"
export LLM_MAX_TOKENS=YOUR_LLM_MAX_TOKENS #e.g. "4000"
export LLM_AUTHORIZATION=YOUR_LLM_AUTHORIZATION #e.g. "Bearer <Your OpenAI API key>"

#Option 2: In case you want to use OpenAIs GPT-3.5-Turbo
export HUGGING_FACE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN #see TL;DR of https://huggingface.co/pyannote/speaker-diarization
export OPENAI_API_KEY=YOUR_KEY #see https://platform.openai.com/account/api-keys

#4 run Meminto 

#For Option 1 (arbitraty LLM)
python main.py -f <file-path> #replace '<file-path>' with path to audio file 

#For Option 2 (GPT-3.5-Turbo)
python main.py -f <file-path> --openai #replace '<file-path>' with path to audio file 
```

### Detailed description 

#### Meminto repository
Clone the [Meminto](#https://github.com/FlorianSchepers/Meminto.git) repository by running <br>
```shell
git clone https://github.com/FlorianSchepers/Meminto.git
```
and then move to its top level folder
```shell
cd Meminto
```
#### Install requirements
I recommend to use a Python version >=3.10.<br>
Next, install the requirements from `requirements.txt`. I recommend to use some kind of virtual environment. You could for example use `pipenv` (https://pypi.org/project/pipenv/):<br>
```shell
pip install --user pipenv #Install pipenv
pipenv install #Create an new environment
pipenv shell #Activate the new environment
pip install -r requirements.txt #Install the requirements
```
#### Setup environment variables

##### Hugging Face Access Token

In order to download a pretrained `pyannote.audio` model for speaker diarization from Hugging Face you will need to accept their terms and get a Hugging Face access token. To do so follow the first three steps of the `TL;DR` at https://huggingface.co/pyannote/speaker-diarization.<br>
Before running Meminto, store your access token in an environment variable called `HUGGING_FACE_ACCESS_TOKEN`.<br>
<br>
On Linux/MacOS:
```Shell
export HUGGING_FACE_ACCESS_TOKEN=YOUR_ACCESS_TOKEN
```
On Windows Powershell:
```Powershell
$Env:HUGGING_FACE_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN" #Windows PowerShell
```
**Note:** The Hugging face access token is required to be set for both of the options described below .

##### Option 1: Required environment variables for an arbitrary LLM 
In order to ensure privacy, you should choose an LLM instance you trust. This could be a local instance or an instance e.g. run by your company. In order for Meminto to communicate with the LLM of your choice, you will need set the following environment variables before running Meminto:<br>
<br>
On Linux/MacOS:
```shell
export LLM_URL=YOUR_LLM_URL #e.g. "https://api.openai.com/v1/chat/completions" for openAI
export LLM_MODEL=YOUR_LLM_MODEL #e.g. "gpt-3.5-turbo"
export LLM_MAX_TOKENS=YOUR_LLM_MAX_TOKENS #e.g. "4000"
export LLM_AUTHORIZATION=YOUR_LLM_AUTHORIZATION #e.g. "Bearer <Your OpenAI API key>"
```
On Windows PowerShell:
```Powershell
$Env:LLM_URL = "YOUR_LLM_URL" #e.g. "https://api.openai.com/v1/chat/completions" for openAI
$Env:LLM_MODEL = "YOUR_LLM_MODEL" #e.g. "gpt-3.5-turbo"
$Env:LLM_MAX_TOKENS = "YOUR_LLM_MAX_TOKENS" #e.g. "4000"
$Env:LLM_AUTHORIZATION= "YOUR_LLM_AUTHORIZATION" #e.g. "Bearer <Your OpenAI API key>"
```

##### Option 2: Required environment variables for using OpenAIs GPT-3.5-Turbo 
As an alternative to Option 1 you can simply use Meminto's pre-configuration for the OpenAI API of GPT-3.5-Turbo. In this case, you only need to store your OpenAI API key in the environment variable `OPENAI_API_KEY`. If you do not have one, get it here: https://platform.openai.com/account/api-keys.<br>
<br>
Linux/MacOS:
```Shell
export OPENAI_API_KEY=YOUR_KEY
```
On Windows PowerShell:
```Powershell
$Env:OPENAI_API_KEY = "YOUR_KEY"
```

#### How to run Meminto

##### Option 1 (arbitrary LLM)
From the top level folder of Meminto run:
```shell
python main.py -f <file-path>
```
Where `<file-path>` corresponds to the path of the audio file for which you want to create the meeting minutes. <br>

##### Option 2 (GPT-3.5-Turbo)
If you want to use GPT-3.5-Turbo via the OpenAI API, you can also use the `--openai` flag:
```shell
python main.py -f <file-path> --openai
```
However, keep in mind that for this option you still need to set the `OPENAI_API_KEY` and `HUGGING_FACE_ACCESS_TOKEN` environment variables as described above.

## Example: Scoreboard meeting

### Transcript of `Scoreboard.wav`

start=0.0s stop=1.1s speaker_SPEAKER_01:<br>
and just continue.<br>
start=2.7s stop=22.0s speaker_SPEAKER_01:<br>
Okay, yeah, thank you for joining. We have our meeting today so that we can plan our new high scoreboard for our new game Pegasus. And yeah, basically we want for each game played the achieved score and the player name to be stored and shown in our high scoreboard.<br>
start=24.1s stop=28.8s speaker_SPEAKER_00:<br>
Okay, if we want to do that, we need to somehow get the names of the players.<br>
start=30.3s stop=34.4s speaker_SPEAKER_01:<br>
Okay, true, we could ask the players for their names at the end of the game.<br>
start=35.2s stop=45.2s speaker_SPEAKER_00:<br>
We can do that. However, then there should also be an option for the player to opt out if they do not want their name to be shown on the board.<br>
start=43.6s stop=44.0s speaker_SPEAKER_01:<br>
the name<br>
start=46.4s stop=53.6s speaker_SPEAKER_01:<br>
Yeah, that's probably a good point. I guess legal would like to have that, uh, that feature implemented.<br>
start=54.3s stop=57.7s speaker_SPEAKER_01:<br>
Yeah, we should also ask GleeGlyph there any other...<br>
start=58.3s stop=61.6s speaker_SPEAKER_01:<br>
compliance related things that we should bear in mind.<br>
start=62.9s stop=65.0s speaker_SPEAKER_00:<br>
Or do we want to store the player scores?<br>
start=66.0s stop=72.6s speaker_SPEAKER_01:<br>
Yeah, let's for the beginning start with an SQLite database and we can then later migrate to something more sophisticated.<br>
start=73.8s stop=80.2s speaker_SPEAKER_00:<br>
Okay, but then we might run into long loading times if we store too many scores in the database.<br>
start=81.4s stop=85.3s speaker_SPEAKER_01:<br>
Okay, then let's just limit the score to the 10 best games.<br>
start=81.4s stop=82.2s speaker_SPEAKER_00:<br>
Okay, then...<br>
start=86.3s stop=92.4s speaker_SPEAKER_00:<br>
Only 10? That's a bit extreme. I was thinking of limiting it to 50k entries.<br>
start=94.0s stop=98.0s speaker_SPEAKER_01:<br>
You really think our game will be played more than 50,000 times?<br>
start=98.5s stop=100.8s speaker_SPEAKER_00:<br>
Sure, otherwise I wouldn't have suggested a limit.<br>
start=102.6s stop=118.0s speaker_SPEAKER_01:<br>
Okay, let's say the top 10,000 games are stored and that should be small enough to ensure good performance. And below that, I guess you probably don't want to see your score anyway.<br>
start=118.7s stop=119.2s speaker_SPEAKER_01:<br>
Fine.<br>
start=121.1s stop=124.8s speaker_SPEAKER_01:<br>
We also want to have the tables sortable.<br>
start=125.9s stop=128.1s speaker_SPEAKER_00:<br>
only by score or also by name.<br>
start=129.0s stop=129.5s speaker_SPEAKER_01:<br>
Both would be.<br>
start=129.5s stop=129.9s speaker_SPEAKER_00:<br>
Be good.<br>
start=131.1s stop=136.3s speaker_SPEAKER_00:<br>
We could also implement a search function in order to find the games of a specific player faster.<br>
start=137.2s stop=138.3s speaker_SPEAKER_01:<br>
Yeah, good idea.<br>
start=139.5s stop=145.9s speaker_SPEAKER_01:<br>
Okay, I think we have everything. I can implement the frontend part of the table and talk to legal.<br>
start=146.6s stop=148.8s speaker_SPEAKER_00:<br>
Okay, then I'll take care of the backend part.<br>
start=150.0s stop=155.0s speaker_SPEAKER_01:<br>
Thank you, that was very productive. I think we are done with our meeting.<br>
start=150.0s stop=151.0s speaker_SPEAKER_00:<br>
Thank you!<br>
start=155.0s stop=159.0s speaker_SPEAKER_00:<br>
Great. Uh, slowly. Do you have any plans for tonight?<br>
start=160.8s stop=162.1s speaker_SPEAKER_00:<br>
Ah, no, not yet.<br>
start=163.5s stop=164.5s speaker_SPEAKER_00:<br>
You could go grab a beer.<br>
start=165.7s stop=169.8s speaker_SPEAKER_01:<br>
Sounds good. Let's meet at the Hercules bar at five then.<br>

### Generated Meeting Notes

**Topic:** Planning high scoreboard for Pegasus game

**Key Decisions:**

- Players will be asked for their names at the end of each game, with an option to opt out of showing their name on the scoreboard.
- Legal team wants the opt-out feature implemented to comply with regulations.
- Check with GleeGlyph for any other compliance-related requirements.
- Score and player names will be stored in an SQLite database.
- Limit the scoreboard to the top 10,000 games to ensure good performance.
- Implement sortable tables for both score and name.
- Implement a search function to find games of a specific player faster.

**Assigned Tasks:**

SPEAKER_00:
- Take care of the backend part of implementing the high scoreboard.

SPEAKER_01:
- Implement the frontend part of the table.
- Talk to legal about the opt-out feature.

**Ai suggestions:**
- Consider implementing a feature to reward players with high scores, such as virtual badges or achievements.
- Create a backup system for the database to prevent data loss.
- Test the performance of the high scoreboard with a large number of game entries.


# Sources
This project was created with the help of the tutorial 'Speech Recognition using Transformers in Python' by Abdeladim Fadheli. The source code of the tutorial is published under the MIT license here https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/nlp/speech-recognition-transformers.<br>
<br>
Furthermore, the project uses pyannote.audio which is published under the MIT license and was published here:
- Bredin et al. *pyannote.audio: neural building blocks for speaker diarization*. ICASSP 2020, IEEE International Conference on Acoustics, Speech, and Signal Processing, (2020)
- Bredin et al. *End-to-end speaker segmentation for overlap-aware resegmentation*. Proc. Interspeech 2021, (2021)