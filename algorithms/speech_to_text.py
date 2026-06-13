import os
import speech_recognition as sr

def transcribe_audio(file_path: str) -> tuple:
    try:
        import pocketsphinx  # noqa: F401
    except ImportError:
        return None, "Offline STT unavailable in this environment. Please paste your transcript manually."

    recognizer = sr.Recognizer()

    if not file_path.lower().endswith('.wav'):
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_file(file_path)
            wav_path = file_path.rsplit('.', 1)[0] + '_converted.wav'
            sound.export(wav_path, format='wav')
            file_path = wav_path
        except Exception as e:
            return None, f"Audio conversion failed: {e}"

    try:
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)
        transcript = recognizer.recognize_sphinx(audio)
        return transcript, None
    except sr.UnknownValueError:
        return None, "Could not understand audio. Please paste your transcript manually."
    except Exception as e:
        return None, str(e)
    finally:
        if file_path.endswith('_converted.wav') and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass
