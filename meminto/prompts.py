CONTEXT = """
You are a team assistant and support the team with its daily work.
"""

INSTRUCTIONS_CREATE_MEETING_MINUTES = """
Your task is to create the meeting minutes for the transcript you get handed by the user.
Proceed step-by-step:
1. Read through the text.
2. Extract the goals of the meeting if possible and add them to the minutes under the title **Goals**.
3. Extract all decisions that were discussed and add them to the minutes under the title **Decisions**.
4. Extract tasks that were assigned to a specific person and add them to the minutes under the title **Assigned Tasks**.
5. Under the title **Additional Notes** add those discussion points from the transcript that seem important but didn't 
   make it into one of the above categories.
6. Return the final meeting notes to the user.
"""

SELECT_LANGUAGE = "Always respond in "

EXAMPLE_OUTPUT_INTRO = """

Example output:

"""

EXAMPLE_OUTPUT = """
**Goals:**

- Organise a surprise party for Adam.

**Decisions:**

Suprise:
    - Suprise Adam when he comes back home.
    - Everybody hides in the kitchen until Adam enters the flat.
    - When Adam enters the flat everybody sings Happy Birthday.
Party:
    - Afterward there will be a party with music, cake and soft drinks.
    - The Party should not be longer then 11 o'clock as everbody need to work the next day.

**Assigned Tasks:**

    - No task assigned

**Additional Notes:**

    - The idea to hire a clown for the party was dissmissed as to expensive.

"""


INSTRUCTIONS_MERGE_MEETING_MINUTES = """
Your task is to create the meeting minutes for a transcript.
The meeting transcript was to long to be processed at once.
Therefore, the transcript has been split into multiple sections.
For each section individual meeting minutes have been created in a previous step.
You will be handed the different meeting minutes of all transcript sections in chronological order.
Your task is to create the meeting minutes for the whole meeting based on the meeting minutes of the different sections of the transcript.
Return only the final meeting minutes to the user.
"""

EXAMPLE_INPUT_INTRO = """

Example input:

"""

EXAMPLE_INPUT = """
Section 1
**Goals:**

- Organise a surprise party for Adam.

**Decisions:**

    - Suprise Adam when he comes back home.
    - Everybody hides in the kitchen until Adam enters the apartment.
    - When Adam enters the apartment everybody sings Happy Birthday.

**Assigned Tasks:**

SPEAKER_02: Organises the Key to the apartment of Adam.

**Additional Notes:**

    - The idea to hire a clown for the party was dissmissed as to expensive.

Section 2
**Decisions:**

    - Afterward the suprise there will be a party with music, cake and soft drinks.
    - The Party should not be longer then 11 o'clock as everbody need to work the next day.

**Assigned Tasks:**

SPEAKER_01: Buys Drinks
Speaker_02: Organises music
SPEAKER_00: Buys Snacks
Speaker_03: No task assigned

"""
