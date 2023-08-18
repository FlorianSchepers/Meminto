import pathlib
import click
from audio_processing import split_audio
from decorators import log_time
from diarization import diarize_audio, load_diarization, save_diarization
from helpers import write_text_to_file
from transscript_to_meeting_minutes import transscript_to_meeting_minutes
from transscription import load_transscript, transscript_audio, save_transscript

ALLOWED_INPUT_FILE_TYPE = {".wav"}
EXAMPLE_FILE_PATH = "examples/Scoreboard.wav"

@log_time
def create_meeting_minutes(audio_source, language):
    
    diarization = diarize_audio(audio_source)
    save_diarization(diarization, 'diarization.pkl')
    
    diarization = load_diarization('diarization.pkl')
    audio_sections = split_audio(audio_source, diarization)
    transscript_sections = transscript_audio(audio_sections, language)
    save_transscript(audio_sections, transscript_sections, 'transscript.txt')
    
    transscript = load_transscript('transscript.txt')
    meeting_minutes = transscript_to_meeting_minutes(transscript, language)
    write_text_to_file(meeting_minutes, 'meeting_minutes.txt')
    print(meeting_minutes)

def parse_input_file_path(input_file):
    file_path = pathlib.Path(input_file) if input_file else pathlib.Path(__file__).parent.resolve() / EXAMPLE_FILE_PATH
    if not file_path.suffix in ALLOWED_INPUT_FILE_TYPE:
        raise Exception(f"Invalid input file type. Only one of the following file type are allowed: {', '.join(str(file_type) for file_type in ALLOWED_INPUT_FILE_TYPE)}") 
    return file_path


@click.command()
@click.option('-f', '--input-file')
def main(input_file)->None:
    audio_source = parse_input_file_path(input_file)
    language = "english"
    create_meeting_minutes(audio_source, language)

if __name__ == "__main__":
    main()