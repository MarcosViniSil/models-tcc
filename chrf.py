import sacrebleu

# Define your hypothesis and reference(s)
# SacreBLEU expects lists of strings
hypothesis = "The cat is on the mat."
references = ["On the mat is the cat."]

# Calculate chrF score
chrf = sacrebleu.sentence_chrf(hypothesis, references)

print(f"chrF score: {chrf.score:.2f}")