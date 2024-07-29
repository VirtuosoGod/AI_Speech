from google.cloud import texttospeech
import os
from pydub import AudioSegment

# Set your Google Cloud credentials file path here
# Make sure to replace the placeholder with the path to your credentials file
# Example: os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/path/to/your/credentials.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "your-google-cloud-credentials.json"

EMOTIONS = {
    'Happy': ['cheerful', 'delighted', 'ecstatic', 'elated', 'encouraged', 'enthusiastic', 'excited',
              'exuberant', 'fulfilled', 'glad', 'good', 'grateful', 'gratified', 'hopeful', 'joyful',
              'jubilant', 'marvelous', 'optimistic', 'peaceful', 'pleased', 'proud', 'relaxed',
              'relieved', 'resolved', 'respected', 'satisfied', 'terrific', 'thrilled', 'tranquil',
              'valued', 'happy'],
    'Sad': ['sad', 'ashamed', 'burdened', 'anguished', 'dejected', 'demoralized', 'deserted',
            'disgusted', 'disheartened', 'dismal', 'heartbroken', 'lonely', 'melancholy', 'miserable',
            'mournful', 'regretful', 'sorrowful'],
    'Angry': ['aggravated', 'annoyed', 'appalled', 'betrayed', 'cheated', 'controlled', 'defeated',
              'disappointed', 'enraged', 'exploited', 'fuming', 'furious', 'harassed', 'hostile',
              'incensed', 'infuriated', 'irritated', 'mad', 'pissed off', 'provoked', 'seething',
              'spiteful', 'vengeful', 'vindictive', 'angry'],
    'Scared': ['alarmed', 'apprehensive', 'afraid', 'anxious', 'bewildered', 'distraught', 'fearful',
               'horrified', 'intimidated', 'nervous', 'panicky', 'petrified', 'terrified', 'threatened',
               'timid', 'scared'],
    'Confused': ['baffled', 'bewildered', 'confused', 'distracted', 'doubtful', 'foggy', 'flustered',
                 'hesitant', 'perplexed', 'puzzled', 'uncertain', 'unclear'],
    'Ambivalent': ['ambivalent', 'conflicted', 'torn', 'uncertain'],
    'Alienated': ['alienated', 'distant', 'neglected', 'uncared for', 'unloved', 'unwanted']
}

def detect_emotion(text):

    text_lower = text.lower()
    for emotion, keywords in EMOTIONS.items():
        if any(keyword in text_lower for keyword in keywords):
            return emotion
    return "neutral"  

def synthesize_speech_chunk(chunk, emotion="neutral", chunk_index=1):

    client = texttospeech.TextToSpeechClient()

    pitch_rate_map = {
        'Happy': ("+5st", "1.2"),
        'Sad': ("-5st", "0.9"),
        'Angry': ("+2st", "1.0"),
        'Scared': ("-2st", "0.9"),
        'Confused': ("0st", "1.0"),
        'Ambivalent': ("0st", "1.0"),
        'Alienated': ("-3st", "0.9")
    }
    
    pitch, rate = pitch_rate_map.get(emotion, ("0st", "1.0"))

    ssml = f"""
    <speak>
        <voice name="en-US-Wavenet-D">
            <prosody pitch="{pitch}" rate="{rate}">
                {chunk}
            </prosody>
        </voice>
    </speak>
    """

    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", name="en-US-Wavenet-D"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    try:
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        filename = f"chunk_{chunk_index}.mp3"
        with open(filename, "wb") as out:
            out.write(response.audio_content)
        return filename
    except Exception as e:
        print(f"An error occurred while synthesizing the chunk: {e}")
    return None

def split_text(text, max_chunk_size=4000):

    chunks = []
    while len(text) > max_chunk_size:
        split_index = text.rfind(' ', 0, max_chunk_size)
        if split_index == -1:
            split_index = max_chunk_size
        
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()
    
    if text:
        chunks.append(text)
    
    return chunks

def combine_mp3_files(file_list, output_file):

    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_mp3(file)
        combined += audio
    
    combined.export(output_file, format="mp3")
    
    for file in file_list:
        os.remove(file)
    print(f"Combined audio content written to file '{output_file}'")
    