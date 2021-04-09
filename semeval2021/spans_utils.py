from IPython.core.display import display, HTML
from http.client import InvalidURL
import matplotlib
import matplotlib.cm
import unicodedata

from collections import defaultdict


PUNCT = '.,?!();:'


def display_spans(spans, text):
    # todo: use style="background-color: #Oxffffff"
    result = []
    spans = set(spans)
    toxic, prev_toxic = False, False
    for i, c in enumerate(text):
        if i in spans:
            toxic = True
            if not prev_toxic:
                result.append('<b>')
        else:
            toxic = False
            if prev_toxic:
                result.append('</b>')
        result.append(c)
        prev_toxic = toxic
    try:
        display(HTML(''.join(result)))
    except InvalidURL:
        print(''.join(result))

        
def display_token_scores(tokens, scores, b=None, cmap=None, bert=False):
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('bwr')
    if b is None:
        b = [0] * len(tokens)

    spans  = []
    for token, score, boldness in zip(tokens, scores, b):
        text = token.replace("Ġ", " ")
        if bert:
            if text.startswith('##'):
                text = text[2:]
            elif text not in PUNCT or text == '(':
                text = ' ' + text
        if boldness:
            text = '<b>{}</b>'.format(text)
        spans.append(
            '<span style="background-color: {}">{}</span>'.format(
                "#{0:02x}{1:02x}{2:02x}".format(*cmap(score, bytes=True)[:3]),
                text
            )
        )
    display(HTML(''.join(spans)))

        
def spans2labels(text, spans, tokenizer, bos=True, left_space=False):
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    char2tok = [0] * len(text)
    left = 0
    first = 1 if bos else 0
    for i, tok in enumerate(tokens[first:-1]):
        right = left + len(tok)
        if i == 0 and left_space:
            right -= 1
        char2tok[left:right] = [i+bos] * (right - left)
        left = right
    labels = [0] * len(tokens)
    for toxic_char in spans:
        labels[char2tok[toxic_char]] = 1
    return labels


def labels2spans(text, labels, tokenizer, space = 'Ġ', bos=True, left_space=False):
    result = []
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    left = 0
    prev_label = 0
    first = 1 if bos else 0
    for i, (tok, label) in enumerate(zip(tokens[first:-1], labels[first:-1])):
        right = left + len(tok)
        if i == 0 and left_space:
            right -= 1
        if label:
            if tok[0] == space and not prev_label:
                left += 1
            result.extend(range(left, right))
        left = right
        prev_label = label
    return result


def decode_spans(text, target_proba, threshold, tokenizer, agg=max, space='Ġ', punct=PUNCT, return_spans=True, truncate=True):
    # try to label a whole multitoken word consistently
    punct = set(punct)
    result = []
    token_labels = [0]
    token_ids = tokenizer.encode(text)
    if truncate:
        token_ids = token_ids[:tokenizer.model_max_length]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    assert len(tokens) <= len(target_proba), '{} > {}'.format(len(tokens), len(target_proba))
    left = 0
    prev_label = 0
    word_start = 0
    word_scores = []
    for i, (tok, tp) in enumerate(zip(tokens[1:], target_proba[1:])):
        right = left + len(tok)
        # start of word (except the first one)
        if i > 0 and (tok[0] == space or tok == '</s>' or tok in punct):
            word_score = agg(word_scores)
            if word_score > threshold:
                if not prev_label and i > 0:
                    word_start += 1
                prev_label = 1
                result.extend(range(word_start, left))
                token_labels.extend([1] * len(word_scores))
            else:
                prev_label = 0
                token_labels.extend([0] * len(word_scores))
            word_scores = []
            word_start = left
        word_scores.append(tp)
        left = right
    token_labels.extend([0] * len(word_scores))
    if return_spans:
        return result
    else:
        return token_labels


def bertlike_normalize(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text.lower())
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def bert_char2tok(text, tokens, bos=1):
    # find approximate positions of each sentencepiece token in a text
    # spaces are all mapped to zero token
    char2tok = [0] * len(text)
    tnorm = bertlike_normalize(text)

    left = 0
    first = int(bos)
    for i, tok in enumerate(tokens[first:-1]):
        t2 = tok.lstrip('##')
        position = tnorm.find(t2, left)
        if position == -1:
            print(t2, 'not found in', tnorm[left:])
            continue
        left = position
        right = left + len(t2)
        if i == 0:
            right -= 1
        char2tok[left:right] = [i+bos] * (right - left)
        left = right
    return char2tok

def bert_spans2labels(text, spans, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    labels = [0] * len(tokens)
    char2tok = bert_char2tok(text, tokens)
    for ch in spans:
        if ch >= len(char2tok):
            # this behavior may happen with weird tokens (e.g. accents), let's just ignore it
            break
        t = char2tok[ch]
        if t > 0:
            labels[t] = 1
    return labels


def bert_labels2spans(text, labels, tokenizer):
    result = []
    token_ids = tokenizer.encode(text)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    c2t = bert_char2tok(text, tokens)
    t2c = defaultdict(list)
    for i, t in enumerate(c2t):
        if t > 0: t2c[t].append(i)
    
    left = 0
    prev_label = 0
    first = 1
    for i, (tok, label) in enumerate(zip(tokens[first:-1], labels[first:-1])):
        token_range = t2c[i+first]
        if not token_range:
            continue
        left, right = min(token_range), max(token_range)
        # print(i, tok, "'" + text[left:right] + "'")
        if label:
            result.extend(range(left - prev_label, right + 1))
        prev_label = label
    return result
