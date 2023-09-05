import os
import requests
from decorators import log_time
from llm import get_chat_completion


INSTRUCTIONS = "You are a team assistant and support the team with its daily work.\n\
You will be handed the transcript of a meeting by the user.\n\
Your task is to create meeting minutes from the transcript.\n\
Do not simply summarize the meeting, but extract the key decisions that were discussed.\n\
The meeting minutes should have a headline that represents the general topic of the discussion.\n\
The meetings minutes should have a section where the key decisions are listed.\n\
If necessary assign tasks to the individual speakers.\n"

SUGGESTIONS_BY_AI = "Add suggestions if the team might have forgotten something.\n"

SELECT_LANGUAGE = "Always respond in "

INTRO_EXAMPLES = "\n\
Example meeting minutes:\n\
\n"

EXAMPLE_1 = "**Topic:** Surprise Party Adam\n\
\n\
**Key Decisions:**\n\
\n\
Suprise:\n\
    - Suprise Adam when he comes back home.\n\
    - Everybody hides in the kitchen until Adam enters the flat.\n\
    - When Adam enters the flat everybody sings Happy Birthday.\n\
Party:\n\
    - Afterward there will be a party with music, cake and soft drinks.\n\
    - The Party should not be longer then 11 o'clock as everbody need to work the next day.\n\
\n\
**Assigned Tasks:**\n\
\n\
SPEAKER_00:\n\
    - Organises the Key to the flat of Adam.\n\
    - Buys Snacks\n\
SPEAKER_01:\n\
    - Buys Drinks\n\
Speaker_02:\n\
    - Organises music\n"

EXAMPLE_1_AI_SUGGESTIONS = "\n\
**AI suggestions:**\n\
    - Who organises a present for Adam?\n"


@log_time
def transcript_to_meeting_minutes(transcript, language, openai):
    system_prompt = (
        INSTRUCTIONS
        + SUGGESTIONS_BY_AI
        + SELECT_LANGUAGE
        + language
        + ".\n"
        + INTRO_EXAMPLES
        + EXAMPLE_1
        + EXAMPLE_1_AI_SUGGESTIONS
    )

    return get_chat_completion(system_prompt, transcript, openai)
