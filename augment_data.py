import random
from pathlib import Path
import csv

from data_loader import load_spam_csv

random.seed(42)

SPAM_PHRASES = [
    "win a free", "claim your prize", "limited time offer", "act now",
    "congratulations", "free entry", "click here", "buy now", "winner",
]

def swap_words(s):
    words = s.split()
    if len(words) < 2:
        return s
    i = random.randint(0, len(words)-2)
    words[i], words[i+1] = words[i+1], words[i]
    return ' '.join(words)

def delete_random_word(s):
    words = s.split()
    if len(words) <= 1:
        return s
    i = random.randrange(len(words))
    del words[i]
    return ' '.join(words)

def duplicate_word(s):
    words = s.split()
    if not words:
        return s
    i = random.randrange(len(words))
    words.insert(i+1, words[i])
    return ' '.join(words)

def append_phrase(s, phrases):
    return s + ' ' + random.choice(phrases)


def augment_text(text, label):
    # apply different strategies for ham/spam
    text = text.strip()
    ops = [swap_words, delete_random_word, duplicate_word]
    # always perform at least one small op to vary the text
    new = text
    # for spam, often append spammy phrase
    if label.lower() == 'spam' and random.random() < 0.6:
        new = append_phrase(new, SPAM_PHRASES)
    # apply 1-2 random ops
    for _ in range(random.randint(1,2)):
        op = random.choice(ops)
        new = op(new)
    # normalize whitespace
    new = ' '.join(new.split())
    return new


def build_augmented_dataset(target_total=8000, out_path='data/spam_aug_8000.csv'):
    df = load_spam_csv()
    # df has columns ['label','text']
    ham = df[df['label'].str.lower() == 'ham']['text'].tolist()
    spam = df[df['label'].str.lower() == 'spam']['text'].tolist()

    if not ham or not spam:
        raise RuntimeError('Need both ham and spam examples in source data')

    target_per_class = target_total // 2
    augmented = []

    # include existing unique rows first (shuffle)
    random.shuffle(ham)
    random.shuffle(spam)

    # add originals (allow duplicates if too few)
    while len(augmented) < target_per_class:
        for t in ham:
            if len(augmented) >= target_per_class:
                break
            augmented.append(('ham', t))
        # if we still don't have enough, augment existing
        if len(augmented) < target_per_class:
            needed = target_per_class - len(augmented)
            for _ in range(needed):
                src = random.choice(ham)
                augmented.append(('ham', augment_text(src, 'ham')))

    ham_aug = augmented[:target_per_class]

    augmented = []
    while len(augmented) < target_per_class:
        for t in spam:
            if len(augmented) >= target_per_class:
                break
            augmented.append(('spam', t))
        if len(augmented) < target_per_class:
            needed = target_per_class - len(augmented)
            for _ in range(needed):
                src = random.choice(spam)
                augmented.append(('spam', augment_text(src, 'spam')))

    spam_aug = augmented[:target_per_class]

    final = ham_aug + spam_aug
    random.shuffle(final)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # write as two columns label,text
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label', 'text'])
        for lbl, txt in final:
            writer.writerow([lbl, txt])

    return out


if __name__ == '__main__':
    p = build_augmented_dataset(8000, 'data/spam_aug_8000.csv')
    print('Augmented dataset written to', p)
