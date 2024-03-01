CONTEXT = "\
You are a team assistant and support the team with its daily work.\n"

INSTRUCTIONS_CREATE_MEETING_MINUTES="\
Your task is to create the meeting minutes for the transcript you get handed by the user.\n\
Proceed step-by-step:\n\
1. Read through the text\n\
2. Extract the goals of the meeting if possible and add them to the minutes under the title **Goals**.\n\
3. Extract all decisions that were discussed and add them to the minutes under the title **Decisions**.\n\
4. Extract tasks that were assigned to a specific person and add them to the minutes under the title **Assigned Tasks**.\n\
5. Under the title **Additional Notes** add those discussion points from the transcript that seem important but didn't make it into one of the above categories.\n\
6. Return the final meeting notes to the user.\n"

SELECT_LANGUAGE = "Always respond in "

EXAMPLE_OUTPUT_INTRO = "\n\
Example output:\n\
\n"
EXAMPLE_OUTPUT = "\
**Goals:**\n\
\n\
- Organise a surprise party for Adam.\n\
\n\
**Decisions:**\n\
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
    - No task assigned\n\
\n\
**Additional Notes:**\n\
\n\
    - The idea to hire a clown for the party was dissmissed as to expensive.\n\
\n"


INSTRUCTIONS_MERGE_MEETING_MINUTES="\
Your task is to create the meeting minutes for a transcript.\n\
The meeting transcript was to long to be processed at once.\n\
Therefore, the transcript has been split into multiple sections.\n\
For each section individual meeting minutes have been created in a previous step.\n\
You will be handed the different meeting minutes of all transcript sections in chronological order.\n\
Your task is to create the meeting minutes for the whole meeting based on the meeting minutes of the different sections of the transcript.\n\
Return only the final meeting minutes to the user.\n"

EXAMPLE_INPUT_INTRO = "\n\
Example input:\n\
\n"
EXAMPLE_INPUT = "\
Section 1\n\
**Goals:**\n\
\n\
- Organise a surprise party for Adam.\n\
\n\
**Decisions:**\n\
\n\
    - Suprise Adam when he comes back home.\n\
    - Everybody hides in the kitchen until Adam enters the apartment.\n\
    - When Adam enters the apartment everybody sings Happy Birthday.\n\
\n\
**Assigned Tasks:**\n\
\n\
SPEAKER_02: Organises the Key to the apartment of Adam.\n\
\n\
**Additional Notes:**\n\
\n\
    - The idea to hire a clown for the party was dissmissed as to expensive.\n\
\n\
Section 2\n\
**Decisions:**\n\
\n\
    - Afterward the suprise there will be a party with music, cake and soft drinks.\n\
    - The Party should not be longer then 11 o'clock as everbody need to work the next day.\n\
\n\
**Assigned Tasks:**\n\
\n\
SPEAKER_01: Buys Drinks\n\
Speaker_02: Organises music\n\
SPEAKER_00: Buys Snacks\n\
Speaker_03: No task assigned\n\
\n"