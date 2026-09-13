"""Keyword detection and color assignment using spaCy NLP."""

import spacy
from core.transcribe import Word


# Lazy-load spaCy model
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def assign_colors(
    words: list[Word],
    green_color: str = "#00FF6A",
    yellow_color: str = "#FFD700",
    white_color: str = "#FFFFFF",
) -> list[Word]:
    """Assign colors to words based on their grammatical role.

    Green: nouns, verbs, named entities, numbers (~60-70%)
    Yellow: 2-4 most impactful words per ~30s (~rare accent)
    White: articles, prepositions, conjunctions, pronouns (~20-30%)
    """
    if not words:
        return words

    nlp = _get_nlp()
    full_text = " ".join(w.text for w in words)
    doc = nlp(full_text)

    # Build a mapping from token index to spaCy token info
    # We need to align spaCy tokens with our word list
    spacy_tokens = list(doc)

    # Create entity span set for quick lookup
    entity_texts = set()
    for ent in doc.ents:
        for token in ent:
            entity_texts.add(token.text.lower())

    # POS tags that get green (important content words only)
    green_pos = {"NOUN", "VERB", "PROPN", "NUM"}
    # POS tags that stay white (function words + modifiers)
    white_pos = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART", "PUNCT", "ADJ", "ADV"}

    # First pass: assign green or white based on POS
    # We align by matching words sequentially to spaCy tokens
    spacy_idx = 0
    for word in words:
        word_lower = word.text.lower().strip(".,!?;:'\"")

        # Find matching spaCy token
        matched = False
        search_range = min(spacy_idx + 5, len(spacy_tokens))
        for j in range(spacy_idx, search_range):
            token = spacy_tokens[j]
            if token.text.lower().strip(".,!?;:'\"") == word_lower:
                if token.text.lower() in entity_texts:
                    word.color = green_color
                elif token.pos_ in green_pos:
                    word.color = green_color
                elif token.pos_ in white_pos:
                    word.color = white_color
                else:
                    word.color = white_color
                spacy_idx = j + 1
                matched = True
                break

        if not matched:
            # Default: if it's a short common word, white; otherwise green
            if len(word_lower) <= 3 and word_lower in {
                "the", "a", "an", "is", "am", "are", "was", "were", "be",
                "to", "of", "in", "on", "at", "by", "for", "and", "or",
                "but", "not", "no", "so", "if", "it", "its", "my", "i",
                "he", "she", "we", "you", "me", "us", "do", "did", "has",
                "had", "can", "may", "will", "as", "up",
            }:
                word.color = white_color
            else:
                word.color = green_color

    # Break streaks of 3+ consecutive green words — demote the shortest
    streak_start = None
    for i, word in enumerate(words):
        is_green = word.color == green_color
        if is_green and streak_start is None:
            streak_start = i
        elif not is_green:
            if streak_start is not None and i - streak_start >= 3:
                streak = list(range(streak_start, i))
                ranked = sorted(streak, key=lambda j: len(words[j].text))
                for j in ranked[: len(streak) - 2]:
                    words[j].color = white_color
            streak_start = None
    if streak_start is not None and len(words) - streak_start >= 3:
        streak = list(range(streak_start, len(words)))
        ranked = sorted(streak, key=lambda j: len(words[j].text))
        for j in ranked[: len(streak) - 2]:
            words[j].color = white_color

    # Second pass: pick yellow accent words (most impactful)
    # ~2-4 yellow words per 30 seconds
    if words:
        total_duration = words[-1].end - words[0].start
        max_yellow = max(2, int(total_duration / 30.0 * 3))

        # Score each green word by "impact" — longer words, entities, lower frequency
        candidates = []
        for i, word in enumerate(words):
            if word.color == green_color:
                score = 0.0
                word_lower = word.text.lower().strip(".,!?;:'\"")

                # Longer words are often more impactful
                score += min(len(word_lower), 10) * 0.5

                # Named entities get a boost
                if word_lower in entity_texts:
                    score += 5.0

                # Numbers/dollar amounts get a boost
                if any(c.isdigit() for c in word.text):
                    score += 4.0

                # Words with high probability are clearer (boost slightly)
                score += word.probability * 1.0

                candidates.append((i, score))

        # Sort by score descending, pick top N with spacing
        candidates.sort(key=lambda x: x[1], reverse=True)

        yellow_indices = set()
        for idx, _score in candidates:
            if len(yellow_indices) >= max_yellow:
                break
            # Ensure yellow words aren't too close together
            too_close = any(abs(idx - yi) < 8 for yi in yellow_indices)
            if not too_close:
                yellow_indices.add(idx)

        for idx in yellow_indices:
            words[idx].color = yellow_color

    return words


def fix_group_contrast(groups, white_color: str = "#FFFFFF") -> list:
    """Ensure visual contrast within caption groups.

    Rules:
    - In a 2-word group where one word is white (function word) and the
      other is colored, split into two single-word groups so the colored
      word is highlighted alone (e.g. "so sure" → ["so"], ["sure"]).
    - In a 2-word group where both are colored, demote the less important
      one to white.
    - Never have yellow directly adjacent to green — looks cluttered.

    Returns the (possibly expanded) list of groups.
    """
    from core.transcribe import WordGroup

    result = []
    for group in groups:
        words = group.words
        if len(words) == 2:
            colored = [w for w in words if w.color != white_color]
            if len(colored) == 1:
                # One function word + one content word → split so the
                # content word is highlighted on its own.
                for w in words:
                    g = WordGroup()
                    g.words = [w]
                    result.append(g)
                continue
            elif len(colored) == 2:
                # Keep the more "important" one (yellow > green, longer > shorter)
                def _rank(w):
                    return (0 if w.color == white_color else 2 if "FFD700" in w.color.upper() else 1, len(w.text))
                colored.sort(key=_rank, reverse=True)
                colored[1].color = white_color
        elif len(words) >= 3:
            # Prevent yellow directly adjacent to green
            for i in range(len(words) - 1):
                a, b = words[i], words[i + 1]
                colors = {a.color.upper(), b.color.upper()}
                if "#FFD700" in colors and "#00FF6A" in colors:
                    shorter = a if len(a.text) <= len(b.text) else b
                    shorter.color = white_color
        result.append(group)
    return result
