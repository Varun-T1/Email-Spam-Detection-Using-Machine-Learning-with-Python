class DummyVectorizer:
    def transform(self, texts):
        return texts

class DummyModel:
    def predict(self, X):
        preds = []
        # simple keyword-based heuristic for demo purposes
        keywords = {
            'free', 'winner', 'win', 'prize', 'claim', 'ticket', 'offer',
            'cash', 'entry', 'congrat', 'buy now', 'click', 'subscribe'
        }
        for x in X:
            txt = x if isinstance(x, str) else ' '.join(x)
            low = txt.lower()
            # explicit override: "not spam" should be ham
            if 'not spam' in low:
                preds.append(0)
                continue
            # if the literal word 'spam' appears (and not in a negation), mark spam
            if 'spam' in low:
                preds.append(1)
                continue
            # count keyword occurrences
            hits = sum(1 for k in keywords if k in low)
            preds.append(1 if hits >= 1 else 0)
        return preds
