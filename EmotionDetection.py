import re
import spacy

nlp = spacy.load("en_core_web_sm")

EMOTIONS = {
    'Happy': [
        'cheerful', 'delighted', 'ecstatic', 'elated', 'encouraged', 'enthusiastic', 'excited',
        'exuberant', 'fulfilled', 'glad', 'good', 'grateful', 'gratified', 'hopeful', 'joyful',
        'jubilant', 'marvelous', 'optimistic', 'peaceful', 'pleased', 'proud', 'relaxed',
        'relieved', 'resolved', 'respected', 'satisfied', 'terrific', 'thrilled', 'tranquil',
        'valued', 'happy'
    ],
    'Sad': [
        'sad', 'ashamed', 'burdened', 'anguished', 'dejected', 'demoralized', 'deserted',
        'disgusted', 'disheartened', 'dismal', 'heartbroken', 'lonely', 'melancholy', 'miserable',
        'mournful', 'regretful', 'sorrowful', 'sad'
    ],
    'Angry': [
        'aggravated', 'annoyed', 'appalled', 'betrayed', 'cheated', 'controlled', 'defeated',
        'disappointed', 'enraged', 'exploited', 'fuming', 'furious', 'harassed', 'hostile',
        'incensed', 'infuriated', 'irritated', 'mad', 'pissed off', 'provoked', 'seething',
        'spiteful', 'vengeful', 'vindictive', 'angry'
    ],
    'Scared': [
        'alarmed', 'apprehensive', 'afraid', 'anxious', 'bewildered', 'distraught', 'fearful',
        'horrified', 'intimidated', 'nervous', 'panicky', 'petrified', 'terrified', 'threatened',
        'timid', 'scared'
    ],
    'Confused': [
        'baffled', 'bewildered', 'confused', 'distracted', 'doubtful', 'foggy', 'flustered',
        'hesitant', 'perplexed', 'puzzled', 'uncertain', 'unclear', 'confused'
    ],
    'Ambivalent': [
        'ambivalent', 'conflicted', 'torn', 'uncertain', 'ambivalent'
    ],
    'Alienated': [
        'alienated', 'distant', 'neglected', 'uncared for', 'unloved', 'unwanted', 'alienated'
    ]
}

def lemmatize_text(text):

    doc = nlp(text)
    lemmatized_text = ' '.join(token.lemma_ for token in doc)
    return lemmatized_text

def extract_emotional_words(text):

    text = lemmatize_text(text)  
    words = set(text.lower().split())  
    emotion_words_found = {category: [] for category in EMOTIONS}
    for category, synonyms in EMOTIONS.items():
        synonyms_set = set(synonyms)  
        matched_words = words & synonyms_set
        emotion_words_found[category] = list(matched_words)
    return emotion_words_found

def clean_text(text):

    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'[^\w\s,.\-\'\"()!?]', '', text)  
    return text
