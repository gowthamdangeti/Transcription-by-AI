import os
import google.generativeai as genai
from pydub import AudioSegment


AudioSegment.converter = r"D:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

# Gemini Transcriber class
class GeminiTranscriber:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.supported_formats = {
            '.mp3': 'audio/mp3',
            '.wav': 'audio/wav',
            '.ogg': 'audio/ogg',
            '.mp4': 'audio/mp4',
        }
    def transcribe(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {ext}")
        with open(file_path, 'rb') as f:
            audio_bytes = f.read()
        response = self.model.generate_content([
            "Transcribe this audio to text:",
            {
                "mime_type": self.supported_formats[ext],
                "data": audio_bytes
            }
        ])
        return response.text.strip()

# Text Translator class
class TextTranslator:
    def __init__(self, api_key=None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
    def translate(self, text, target_lang_name, target_lang_code):
        prompt = f"""
You are a professional translator.
Translate the following text ONLY into {target_lang_name} language.
- Do NOT include the original text.
- Output ONLY the translated text.
- Use the proper script for {target_lang_name}.

Text:
{text}
"""
        response = self.model.generate_content(prompt)
        return response.text.strip()

# Audio splitter utility
def split_audio(file_path, chunk_length_ms=10 * 60 * 1000):
    audio = AudioSegment.from_file(file_path)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_path = f"data/chunk_{i // chunk_length_ms}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)
    return chunks

# Main function
def main():
    data_dir = "data"
    transcript_dir = "outputs/transcripts"
    translation_dir = "outputs/translations"
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(translation_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(data_dir)
                   if os.path.splitext(f)[1].lower() in ['.mp3', '.wav', '.mp4', '.ogg']]
    if not audio_files:
        print(f"No audio files found in '{data_dir}' folder. Please add audio files.")
        return

    print("Available audio files:")
    for i, f in enumerate(audio_files, 1):
        print(f"  {i}. {f}")
    choice = input(f"Select file to transcribe (1-{len(audio_files)}): ").strip()
    try:
        audio_file = audio_files[int(choice) - 1]
    except Exception:
        print("Invalid choice.")
        return
    audio_path = os.path.join(data_dir, audio_file)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()

    transcriber = GeminiTranscriber(api_key)
    translator = TextTranslator(api_key)

    print(f"\nSplitting '{audio_file}' into smaller chunks...")
    chunks = split_audio(audio_path)
    print(f"Created {len(chunks)} chunks.")

    full_transcript = ""
    for i, chunk_path in enumerate(chunks, 1):
        print(f"\nTranscribing chunk {i}/{len(chunks)}: {chunk_path}")
        transcript_part = transcriber.transcribe(chunk_path)
        full_transcript += f"\n\n--- Chunk {i} ---\n{transcript_part}"

    transcript_file = os.path.join(transcript_dir, "transcript.txt")
    with open(transcript_file, "w", encoding="utf-8") as f:
        f.write(full_transcript)
    print(f"\n Full transcript saved to '{transcript_file}'")

    languages = [
        ("English", "en"), ("Hindi", "hi"), ("Telugu", "te"),
        ("Spanish", "es"), ("French", "fr"), ("German", "de"),
        ("Chinese (Simplified)", "zh"), ("Japanese", "ja"),
        ("Arabic", "ar"), ("Russian", "ru")
    ]
    print("\nSupported target languages:")
    for i, (name, _) in enumerate(languages, 1):
        print(f"  {i}. {name}")

    lang_input = input("Enter language numbers separated by commas (e.g. 1,2,3): ")
    lang_indices = [int(x.strip()) - 1 for x in lang_input.split(",") if x.strip().isdigit()]

    for idx in lang_indices:
        lang_name, lang_code = languages[idx]
        print(f"\nTranslating transcript to '{lang_name}' ({lang_code}) ...")
        translation = translator.translate(full_transcript, lang_name, lang_code)

        translation_file = os.path.join(translation_dir, f"translation_{lang_code}.txt")
        with open(translation_file, "w", encoding="utf-8") as f:
            f.write(translation)
        print(f"Translation saved to '{translation_file}'")

        preview_len = 500
        print(f"\n--- Translation Preview ({lang_code}) ---")
        print(translation[:preview_len])
        if len(translation) > preview_len:
            print("...")

if __name__ == "__main__":
    main()
