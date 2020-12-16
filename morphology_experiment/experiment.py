from load_texts import TextLoader
from base_classifier import BaseClassifier

print("Pre-processing...")
TextLoader().load_texts()

# print("Evaluating Plaintext...")
# plaintext_classifier = BaseClassifier(method="plaintext", batch_size=32, learning_rate=0.0001, epochs=1, embedding_size=64, hidden_size=32)
# plaintext_classifier.run_bootstrap(2)

# print("Evaluating Lemmatization...")
# lemma_classifier = BaseClassifier(method="lemma", batch_size=32, learning_rate=0.0001, epochs=15, embedding_size=64, hidden_size=32)
# lemma_classifier.run_bootstrap(10)

# print("Evaluating Lemma Concat...")
# lemma_concat_classifier = BaseClassifier(method="lemma_concat", batch_size=64, learning_rate=0.0001, epochs=10, embedding_size=64, hidden_size=32)
# lemma_concat_classifier.run_bootstrap(10)

# print("Evaluating Byte Pair Encoding...")
# bytepair_classifier = BaseClassifier(method="encoded_labeled", batch_size=64, learning_rate=0.0001, epochs=15, embedding_size=64, hidden_size=32)
# bytepair_classifier.run_bootstrap(10)


