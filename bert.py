from bert_score import score

geradoPeloModelo = ["O tempo esta congelante hoje."]
referencia = ["Hoje esta frio."]

PRECISAO, REVOCACAO, F1_SCORE = score(geradoPeloModelo, referencia, lang="pt", verbose=True)

print(F1_SCORE.mean())