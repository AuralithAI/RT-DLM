class DataProcessor:
    def __init__(self):
        # Need to set vocabulary if possible
        self.vocab = None

    def preProcess_Text(self, text):
        # Preprocess the text
        return text
    
    def tokenize(self, text):
        # Tokenize the text
        return text.split()
    
    def convert_text_to_tokens(self, text):
        # Convert text to tokens
        return [self.vocab[word] for word in text]