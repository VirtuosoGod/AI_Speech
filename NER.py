from transformers import pipeline

model_checkpoint = "Wizard007Bond/bert-finetuned-ner"
token_classifier = pipeline("token-classification", model=model_checkpoint, aggregation_strategy="first")

def merge_entities(entities):
    merged_entities = []
    current_entity = None

    for entity in entities:
        if current_entity is None:
            current_entity = entity
        else:
            if (current_entity['entity_group'] == entity['entity_group'] and
                current_entity['word'].endswith(entity['word'][0])):
                current_entity['word'] += ' ' + entity['word']
                current_entity['score'] = max(current_entity['score'], entity['score'])
            else:
                merged_entities.append(current_entity)
                current_entity = entity

    if current_entity is not None:
        merged_entities.append(current_entity)

    return merged_entities

def remove_duplicates(entities):
    seen = set()
    unique_entities = []
    for entity in entities:
        entity_key = (entity['word'].strip().lower(), entity['entity_group'])
        if entity_key not in seen:
            seen.add(entity_key)
            unique_entities.append(entity)
    return unique_entities

def filter_labels(entities, allowed_labels):
    return [entity for entity in entities if entity['entity_group'] in allowed_labels]

def perform_ner(text, chunk_size=2000):
    all_results = []
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    for chunk in chunks:
        try:
            results = token_classifier(chunk)
            all_results.extend(results)
        except Exception as e:
            raise RuntimeError(f"An error occurred during NER processing: {e}")

    merged_results = merge_entities(all_results)
    unique_results = remove_duplicates(merged_results)
    allowed_labels = {'LOC', 'PER'}
    final_results = filter_labels(unique_results, allowed_labels)
    
    per_entities = [entity['word'] for entity in final_results if entity['entity_group'] == 'PER']
    loc_entities = [entity['word'] for entity in final_results if entity['entity_group'] == 'LOC']
    
    return per_entities, loc_entities
