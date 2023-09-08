CONTEXT = "\
You are a team assistant and support the team with its daily work.\n\
You will be handed the transcript of a meeting by the user.\n\
Your task is to create meeting minutes from the transcript.\n"

INSTRUCTIONS_CREATE_MEETING_MINUTES="\
Do not simply summarize the meeting, but extract the key decisions that were discussed.\n\
The meeting minutes should have a headline that represents the general topic of the meeting.\n\
The meetings minutes should have a section where the key decisions are listed.\n\
Do not assign any task to a speaker if the task was not explicitly assigned to the speaker during the meeting.\n"

AI_SUGGESTIONS = "Add suggestions if the team might have forgotten something.\n"

SELECT_LANGUAGE = "Always respond in "

EXAMPLE_INPUT_INTRO = "\n\
Example input:\n\
\n"
EXAMPLE_INPUT = "Part 1\n\
**Meeting Topic:** Surprise Party Adam\n\
\n\
**Key Decisions:**\n\
\n\
Suprise:\n\
    - Suprise Adam when he comes back home.\n\
    - Everybody hides in the kitchen until Adam enters the flat.\n\
    - When Adam enters the flat everybody sings Happy Birthday.\n\
\n\
**Assigned Tasks:**\n\
\n\
SPEAKER_00:\n\
    - Organises the Key to the flat of Adam.\n\
\n\
**AI suggestions:**\n\
    - Who organises a present for Adam?\n\
\n\
Part 2\n\
**Meeting Topic:** Party Adam\n\
\n\
**Key Decisions:**\n\
\n\
Party:\n\
    - Afterward the suprise there will be a party with music, cake and soft drinks.\n\
    - The Party should not be longer then 11 o'clock as everbody need to work the next day.\n\
\n\
**Assigned Tasks:**\n\
\n\
SPEAKER_00:\n\
    - Buys Snacks\n\
SPEAKER_01:\n\
    - Buys Drinks\n\
Speaker_02:\n\
    - Organises music\n\
Speaker_03:\n\
    - No task assigned\n\
\n\
**AI suggestions:**\n\
    - Who invites Adams other friends to the party?\n"

EXAMPLE_INTRO = "\n\
Example output:\n\
\n"

EXAMPLE_1 = "**Meeting Topic:** Surprise Party Adam\n\
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
    - Organises music\n\
Speaker_03:\n\
    - No task assigned\n"

EXAMPLE_1_AI_SUGGESTIONS = "\n\
**AI suggestions:**\n\
    - Who organises a present for Adam?\n\
    - Who invites Adams other friends to the party?\n"

INSTRUCTIONS_MERGE_MEETING_MINUTES="\
The meeting transcript was to long to be processed at once.\n\
It has been split into multiple sections.\n\
For each section individual meeting minutes have been created in a previous step.\n\
You will be handed the different meeting minutes of all sections of the meeting in chronological order.\n\
Your task is to create the meeting minutes for the whole meeting based on the meeting minutes of the different sections of the meeting.\n"