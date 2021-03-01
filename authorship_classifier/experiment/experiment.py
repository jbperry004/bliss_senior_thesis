from base_classifier import BaseClassifier

bytepair_classifier = BaseClassifier(division="sents", batch_size=64, learning_rate=0.0001, epochs=1, embedding_size=64, hidden_size=32)
bytepair_classifier.run_bootstrap(1)