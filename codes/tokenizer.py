import re
from collections import Counter

class Tokenizer:
    def __init__(self, text, subword=False):
        self.text = text
        self.subword = subword
    
    def sentence_tokenize(self):
        """Splits text into sentences based on punctuation."""
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', self.text.strip()) if s]
    
    def word_tokenize(self, sentence):
        """Splits a sentence into words, handling punctuation properly."""
        return re.findall(r"\b\w+(?:'\w+)?\b|[.,!?;]", sentence)
    
    def subword_tokenize(self, word, vocab=None):
        """Breaks words into subwords using a simple Byte Pair Encoding (BPE) approach."""
        if not vocab:
            return list(word)  # Fallback: return characters
        
        subwords = []
        while word:
            match = next((sub for sub in sorted(vocab, key=len, reverse=True) if word.startswith(sub)), None)
            if match:
                subwords.append(match)
                word = word[len(match):]
            else:
                subwords.append(word)
                break
        return subwords
    
    def remove_stopwords(self, words):
        """Removes unnecessary words (stopwords) to optimize query parsing."""
        stopwords = {"is", "a", "the", "this", "it", "if", "and", "to", "let's", "see"}
        return [word for word in words if word.lower() not in stopwords]
    
    def agentic_chunking(self, tokenized_sentences):
        """Groups tokens into meaningful chunks based on semantic propositioning and context."""
        chunks = []
        current_chunk = []
        
        for sentence in tokenized_sentences:
            for word in sentence:
                current_chunk.append(word)
                if word in {'.', '!', '?', ',', ';', ':'}:  # Use punctuation as chunk boundaries
                    chunks.append(current_chunk)
                    current_chunk = []
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
        
        return chunks
    
    def tokenize(self):
        """Tokenizes text into sentences, words, removes stopwords, applies subword tokenization, and chunks."""
        tokenized_output = []
        
        for sentence in self.sentence_tokenize():
            words = self.remove_stopwords(self.word_tokenize(sentence))
            
            if self.subword:
                vocab = self.train_bpe(words)
                words = [subword for word in words for subword in self.subword_tokenize(word, vocab)]
            
            tokenized_output.append(words)
        
        return self.agentic_chunking(tokenized_output)
    
    def train_bpe(self, words, num_merges=10):
        """Trains a simple Byte Pair Encoding (BPE) model."""
        pairs = Counter((word[i], word[i+1]) for word in words for i in range(len(word) - 1))
        vocab = {"".join(word) for word in words}
        
        for _ in range(min(num_merges, len(pairs))):
            most_common = max(pairs, key=pairs.get, default=None)
            if not most_common:
                break
            vocab.add("".join(most_common))
            pairs = Counter({k: v for k, v in pairs.items() if k != most_common})
        
        return vocab

# Example usage
text = "Right engine is on fire and the left engine is not working due to hydraulic failure"
tokenizer = Tokenizer(text, subword=True)
tokens = tokenizer.tokenize()
print(tokens)
