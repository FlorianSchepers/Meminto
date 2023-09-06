from decorators import log_time
from llm import get_chat_completion
from prompts import EXAMPLE_1, EXAMPLE_1_AI_SUGGESTIONS, INSTRUCTIONS, INTRO_EXAMPLES, SELECT_LANGUAGE, SUGGESTIONS_BY_AI

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
