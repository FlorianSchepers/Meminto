from enum import Enum
from pathlib import Path
import pickle
from typing import Any


class Language(Enum):
    GERMAN = "german"
    ENGLISH = "english"


ALLOWED_INPUT_FILE_TYPE = {".wav", ".mp3"}
EXAMPLE_FILE_PATH = "examples/Scoreboard.wav"


def select_language(language: str) -> Language:
    match language:
        case Language.ENGLISH.value:
            return Language.ENGLISH
        case Language.GERMAN.value:
            return Language.GERMAN
        case _:
            return Language.ENGLISH


def parse_input_file_path(input_file: str) -> Path:
    file_path = Path(input_file)

    if not file_path.exists():
        raise Exception(f"Input file does not exist at given file path: {file_path}")
    
    if not file_path.is_file():
        raise Exception(f"Input file path '{file_path}' does not reference a file.")

    if file_path.suffix not in ALLOWED_INPUT_FILE_TYPE:
        raise Exception(
            f"Invalid input file type. Only one of the following file type are allowed: {', '.join(str(file_type) for file_type in ALLOWED_INPUT_FILE_TYPE)}"
        )

    return file_path.resolve()


def parse_output_folder_path(output_folder: str) -> Path:
    output_folder_path = Path(output_folder)

    if not output_folder_path.exists():
        raise Exception(
            f"Output folder does not exist at given folder path: {output_folder}"
        )

    if not output_folder_path.is_dir():
        raise Exception(f"Output folder path '{output_folder_path}' does not reference a folder.")

    return output_folder_path.resolve()


def write_text_to_file(text: str, file_path: Path) -> None:
    with open(file_path, "w") as file:
        file.write(text)


def load_pkl(file_path: Path) -> Any:
    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_as_pkl(content: Any, file_path: Path) -> None:
    with open(file_path, "wb") as file:
        pickle.dump(content, file, pickle.HIGHEST_PROTOCOL)
