"""
A collection of prompts for Whisper speech transcription.
"""
# Reference: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

class WhisperTranscriptionPrompt:
    def __init__(self):
        pass

    def generate_prompt(self, task):
        if not task in ["transcribe", "translate"]:
            raise ValueError("Task must be either 'transcribe' or 'translate'!")
        special_tokens = [
            "<|startoftranscript|>" 
        ]

        # if use_no_time_stamps:
        #     special_tokens.append("<|notimestamps|>")

        special_tokens.append(f"<|{task}|>")

        return "".join(special_tokens)