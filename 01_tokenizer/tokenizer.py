import random

class simple_tokenizer:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.next_index = random.randint(1000, 9999)

    def encode(self, text):
        tokens = []
        for word in text.split():
            if word not in self.word_to_index:
                self.word_to_index[word] = self.next_index
                self.index_to_word[self.next_index] = word
                self.next_index = random.randint(1000,9999)
            tokens.append(self.word_to_index[word])
        return tokens
    
    def decode(self, tokens):
        words=""
        for token in tokens:
            if token in self.index_to_word:
                words += self.index_to_word[token]
        return words
    
tokenizer = simple_tokenizer()
text = "Hello, World!"
encoded = tokenizer.encode(text)
print(f"Encoded: {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")