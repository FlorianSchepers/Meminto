# Meminto - The AI based Meeting Minutes Tool

Meminto is an AI based tool to create meeting minutes. Just hand it a '.wav' audio file of a recorded meeting and it will automatically generate your meeting minutes.<br>
- In a first step it will use speaker-diarization of pyannote.audio in order to differentiate between the different speakers.<br>
- It will then use whisper in order to generate an transcript of the meeting.<br>
- Finally, it will use an LLM to generate the meeting minutes. 

While there are a lot of commercially available tools to generate meeting notes, Meminto was intended to be an open source tool that gives the users control over their data.<br>
Therefore, the diarization and transcription are executed on your local device. Note however, that for the final creation of the meeting minutes the user needs to specify an LLM instance to use (see instructions below). It is in the responsibility of the user to choose an LLM instance with a sufficient degree of confidentiality.

## How to setup and run Meminto

### TL;DR
Step 1. - Clone the Meminto repository
```shell
git clone https://github.com/FlorianSchepers/Meminto.git
cd Meminto
```
Step 2. - Setup virtual environment and install the dependencies
```shell
pipx install poetry
poetry install
poetry shell
```
Step 3. - Define environment variables
   - create a file called `.env` in the top level folder of Meminto
   - open the file and fill in the following information:
```shell 
HUGGING_FACE_ACCESS_TOKEN=<your_access_token> #see TL;DR of https://huggingface.co/pyannote/speaker-diarization
LLM_URL=<your_llm_url> #e.g. "https://api.openai.com/v1/chat/completions" for openAI
LLM_MODEL=<your_llm_model> #e.g. "gpt-3.5-turbo"
LLM_MAX_TOKENS=<yor_llm_max_tokens> #e.g. "4000"
LLM_AUTHORIZATION=<your_llm_authorization> #e.g. "Bearer <Your OpenAI API key>"
```
Step 4. - Run Meminto 
```shell
python meminto/main.py -f <file-path> #replace '<file-path>' with path to audio file 
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
As Python version Python >= 3.10 is recommended.<br>
Meminto uses Poetry for the dependency management.<br>
[Install Poetry](https://python-poetry.org/docs/) and run the following command in the root folder of the project in order to setup and activate the virtual environment
```shell
poetry install
poetry shell
```

#### Setup environment variables
All environment variables that will be used by Meminto can be pre-defined in a local `.env` file in the root level folder of Meminto. If it does not yet exist you will need to create it first.

##### Hugging Face Access Token

In order to download a pretrained `pyannote.audio` model for speaker diarization from Hugging Face you will need to accept their terms and get a Hugging Face access token. To do so follow the first three steps of the `TL;DR` at https://huggingface.co/pyannote/speaker-diarization.<br>
Before running Meminto, write your access token to the `.env` file in the following format:
```shell
HUGGING_FACE_ACCESS_TOKEN=<your_access_token>
```

##### LLM Environment variables 
In order to ensure privacy, you should choose an LLM instance you trust. This could be a local instance or an instance e.g. run by your company. In order to communicate with the LLM of your choice Meminto will need the LLM URL, model, authorization key and max tokens. You can provide this information by adding it to the `.env` file in the following format:
```shell
LLM_URL=<your_llm_url> #e.g. "https://api.openai.com/v1/chat/completions" for openAI
LLM_MODEL=<your_llm_model> #e.g. "gpt-3.5-turbo"
LLM_MAX_TOKENS=<yor_llm_max_tokens> #e.g. "4000"
LLM_AUTHORIZATION=<your_llm_authorization> #e.g. "Bearer <Your OpenAI API key>"
```

#### How to run Meminto

From the top level folder of Meminto run:
```shell
python meminto/main.py -f <file-path>
```
Where `<file-path>` corresponds to the path of the audio file for which you want to create the meeting minutes. There is an example file stored at `examples/Scoreboard.wav`.<br>

## Example: Scoreboard meeting

Location: `examples/Scoreboard.wav`

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