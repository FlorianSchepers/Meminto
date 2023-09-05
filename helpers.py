from enum import Enum
import pathlib
import pickle

class Language(Enum):
    GERMAN = 'german'
    ENGLISH = 'english'

ALLOWED_INPUT_FILE_TYPE = {".wav", ".mp3"}
EXAMPLE_FILE_PATH = "examples/Scoreboard.wav"

def select_language(language):
    match language:
        case Language.ENGLISH.value:
            return Language.ENGLISH.value
        case Language.GERMAN.value:
            return Language.GERMAN.value
        case _:
            return Language.ENGLISH.value
        
def parse_input_file_path(input_file):
    file_path = (
        pathlib.Path(input_file)
        if input_file
        else pathlib.Path(__file__).parent.resolve() / EXAMPLE_FILE_PATH
    )
    if not file_path.suffix in ALLOWED_INPUT_FILE_TYPE:
        raise Exception(
            f"Invalid input file type. Only one of the following file type are allowed: {', '.join(str(file_type) for file_type in ALLOWED_INPUT_FILE_TYPE)}"
        )
    return file_path

def write_text_to_file(text, file_path):
    with open(file_path, "w") as file:
        file.write(text)

def load_pkl(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_as_pkl(content, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(content, file, pickle.HIGHEST_PROTOCOL)
