from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = [['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']]
candidate = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

smooth = SmoothingFunction().method1
score = sentence_bleu(reference, candidate, smoothing_function=smooth)

print(f"BLEU score: {score:.4f}")