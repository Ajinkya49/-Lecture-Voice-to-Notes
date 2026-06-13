import os
import speech_recognition as sr


def transcribe_audio(file_path: str) -> tuple[str | None, str | None]:
    """
    Transcribe an audio file to text using offline PocketSphinx.

    Supports .wav natively.  .mp3 / .m4a are converted to .wav via pydub
    before recognition so that PocketSphinx (which only accepts PCM WAV) can
    process them.

    Returns
    -------
    (transcript, None)   on success
    (None, error_message) on failure
    """
    recognizer = sr.Recognizer()

    # ── Convert non-WAV formats ──────────────────────────────────────────
    if not file_path.lower().endswith('.wav'):
        try:
            from pydub import AudioSegment
            sound = AudioSegment.from_file(file_path)
            wav_path = file_path.rsplit('.', 1)[0] + '_converted.wav'
            sound.export(wav_path, format='wav')
            file_path = wav_path
        except Exception as conv_err:
            return None, f"Audio conversion failed: {conv_err}"

    # ── Transcribe ───────────────────────────────────────────────────────
    try:
        with sr.AudioFile(file_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.record(source)

        transcript = recognizer.recognize_sphinx(audio)
        return transcript, None

    except sr.UnknownValueError:
        return None, (
            "Could not understand the audio.  "
            "Please try a clearer recording or paste the transcript manually."
        )
    except sr.RequestError as req_err:
        return None, f"PocketSphinx engine error: {req_err}"
    except Exception as exc:
        return None, str(exc)
    finally:
        # Clean up auto-generated converted file
        if file_path.endswith('_converted.wav') and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass
