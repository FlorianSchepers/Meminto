from enum import Enum
import torch
from transformers import *
from audio_processing import SAMPLING_RATE, batch
from decorators import log_time


class WHISPER_MODEL_SIZE(Enum):
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def setup_whisper(model_size=WHISPER_MODEL_SIZE.MEDIUM, english_only=False):
    match model_size:
        case WHISPER_MODEL_SIZE.TINY:
            if english_only:
                whisper_model_name = "openai/whisper-tiny.en" # English-only, ~ 151 MB
            else:
                whisper_model_name = "openai/whisper-tiny" # multilingual, ~ 151 MB
        case WHISPER_MODEL_SIZE.BASE:
            if english_only:
                whisper_model_name = "openai/whisper-base.en" # English-only, ~ 290 MB
            else:
                whisper_model_name = "openai/whisper-base" # multilingual, ~ 290 MB
        case WHISPER_MODEL_SIZE.SMALL:
            if english_only:
                whisper_model_name = "openai/whisper-small.en" # English-only, ~ 967 MB
            else:
                whisper_model_name = "openai/whisper-small" # multilingual, ~ 967 MB
        case WHISPER_MODEL_SIZE.MEDIUM:
            if english_only:
                whisper_model_name = "openai/whisper-medium.en" # English-only, ~ 3.06 GB
            else:
                whisper_model_name = "openai/whisper-medium" # multilingual, ~ 3.06 GB                             
        case WHISPER_MODEL_SIZE.LARGE:
            whisper_model_name = "openai/whisper-large-v2" # multilingual, ~ 6.17 GB
        case _:
            whisper_model_name = "openai/whisper-medium" # multilingual, ~ 3.06 GB

    whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)
    return whisper_processor, whisper_model

def get_transcription_whisper(audio, whisper_processor, whisper_model, language="english", skip_special_tokens=True):
    audio_batched = batch(audio, SAMPLING_RATE*30)
    transcription = []
    for section in audio_batched:
        input_features = whisper_processor(section, return_tensors="pt", sampling_rate=SAMPLING_RATE).input_features.to(device)
        forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language=language, task="transcribe")
        predicted_ids = whisper_model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription.append(whisper_processor.batch_decode(predicted_ids, skip_special_tokens=skip_special_tokens)[0])
    return transcription

@log_time
def transscript_audio(audio_sections, language):
    print("Creating transscript from diarized audio")
    section_transscripts = []
    (whisper_processor, whisper_model) = setup_whisper(model_size=WHISPER_MODEL_SIZE.MEDIUM)
    total_number_of_sections = len(audio_sections)
    for section_number, section in enumerate(audio_sections):
        print(f"Transscribing section {section_number} of {total_number_of_sections}.")
        section_transscripts.append(
            get_transcription_whisper(
                section["audio"],  
                whisper_processor=whisper_processor,
                whisper_model=whisper_model, 
                language=language, 
                skip_special_tokens=True
            )
        )
    return section_transscripts

def save_transscript(audio_sections, section_transscripts, file_path):
    with open(file_path, 'w') as file:
        for idx, section in enumerate(audio_sections):
            file.write(f"start={section['turn'].start:.1f}s stop={section['turn'].end:.1f}s speaker_{section['speaker']}:\n")
            for batch in section_transscripts[idx]:
                file.write(batch.strip())
            file.write("\n")

def load_transscript(file_path):
    with open(file_path, 'r') as file:     
        return file.read()