import re


def text_cleaner(text):
    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        {r'</(div)\s*>\s*': u'\n'},  # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},  # newline after </p> and </div> and <h1/>...
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},  # show links instead of texts
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''},  # remove spaces at the beginning
    ]
    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
    text = text.translate({ord(c): " " for c in "[@_!#$%^&*()<>?\|}{~:.,;0123456789°ºð“ø.®åÿ¥öï—›®+º«»©¶ß=’²\']/"})
    return text.lower()


def fix_tokens(tokens, STOPWORDS):
    tokens_end = []
    for item in tokens:
        if len(item) == 1 or len(item) > 15:
            continue
        if item in STOPWORDS:
            continue
        else:
            tokens_end.append(item)
    return tokens_end