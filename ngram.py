import sys
import os
import re
import pandas as pd
from collections import defaultdict, Counter
import random
import math
from sklearn.model_selection import train_test_split
import javalang


def preprocess_code(code):
    tokens = list(javalang.tokenizer.tokenize(code))
    token_strings = [str(token.value) for token in tokens]
    return token_strings

class NgramModel:
    def __init__(self, n, k=0.1):
        self.n = n
        self.k = k
        self.ngrams = defaultdict(Counter)
        self.context_totals = defaultdict(int)
        self.vocab = set()
        self.vocab_size = 0

    def train(self, corpus):
        for tokens in corpus:
            padded_tokens = ['<s>'] * (self.n - 1) + tokens + ['</s>']
            for token in tokens:
                self.vocab.add(token)  # Build vocabulary
            for i in range(len(padded_tokens) - self.n + 1):
                context = tuple(padded_tokens[i:i+self.n-1])
                token = padded_tokens[i+self.n-1]
                self.ngrams[context][token] += 1
                self.context_totals[context] += 1
        self.vocab_size = len(self.vocab)

    def get_prob(self, context, token):
        """ Gets the probability of a token given a context using K-smoothing """
        context = tuple(context)
        token_count = self.ngrams[context][token] if context in self.ngrams else 0
        context_count = self.context_totals[context] if context in self.context_totals else 0
        
        # K-Smoothing
        return (token_count + self.k) / (context_count + self.k * self.vocab_size)
    
    def unk_prob(self, context):
        return self.get_prob(context, '<unk>')

    def generate_code(self, context, max_length=50):
        """ Generates code based on a context """
        context = ['<s>'] * (self.n - 1) + context
        generated = []
        probs = []

        for _ in range(max_length):
            context_tuple = tuple(context[-(self.n-1):])
            
            if context_tuple not in self.ngrams:
                next_token = '<unk>'
                prob_dist = [self.unk_prob(context_tuple)]
            else:
                tokens, counts = zip(*self.ngrams[context_tuple].items())
                prob_dist = [self.get_prob(context_tuple, token) for token in tokens]
                tokens = list(tokens) + ['<unk>']
                prob_dist = list(prob_dist) + [self.unk_prob(context_tuple)]
                next_token = random.choices(tokens, weights=prob_dist, k=1)[0]
            
            if next_token == '</s>':
                break
            
            generated.append(next_token)
            if next_token == '<unk>':
                probs.append(self.unk_prob(context_tuple))
            else:
                probs.append(self.get_prob(context_tuple, next_token))
            context.append(next_token)
        
        return ' '.join(generated), probs


#Evaluate the Model using Perplexity
def calculate_perplexity(model, corpus):
    """ Calculates perplexity for the given corpus using smoothed probabilities """
    log_prob_sum = 0
    word_count = 0
    
    for tokens in corpus:
        padded_tokens = ['<s>'] * (model.n - 1) + tokens + ['</s>']
        for i in range(model.n - 1, len(padded_tokens)):
            context = padded_tokens[i-model.n+1:i]
            token = padded_tokens[i]
            if token not in model.vocab:
                token = '<unk>'
            prob = model.get_prob(context, token)
            log_prob_sum += math.log(prob if prob > 0 else 1e-10)  # Avoid log(0)
            word_count += 1
            
    perplexity = math.exp(-log_prob_sum / word_count)
    return perplexity

#Load Java Corpus from CSV
def load_and_split_data(file_path, column_name, test_size=0.2):
    """ Loads the Java methods from a CSV file and splits into training and testing sets """
    df = pd.read_csv(file_path)
    methods = df[column_name].dropna()
    train_methods, test_methods = train_test_split(methods, test_size=test_size, shuffle=True, random_state=42)
    train_corpus = [preprocess_code(method) for method in train_methods]
    test_corpus = [preprocess_code(method) for method in test_methods]

    return train_corpus, test_corpus
    
def output(model, test_corpus, output_file_path, num_samples=100, max_length=50):
    """ Writes generated code from the N-gram model to a CSV file """
    results = []
    for i, test_method in enumerate(test_corpus[:num_samples]):
        context = test_method[-(model.n-1):]  # Use the last (n-1) tokens as the context
        generated_code, probs = model.generate_code(context, max_length)
        results.append({
            'input_context': ' '.join(test_method),
            'generated_code': generated_code,
            'probabilities': probs
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, index=False)
    
def main(csv_file_path, column_name, n=3, test_size=0.2, output_file_path='generated_codes.csv', num_samples=100, k=0.1):
    # Load and split data
    train_corpus, test_corpus = load_and_split_data(csv_file_path, column_name, test_size)

    # Initialize and train the N-gram model
    ngram_model = NgramModel(n, k=k)
    ngram_model.train(train_corpus)

    # Calculate perplexity on the training and test data
    train_perplexity = calculate_perplexity(ngram_model, train_corpus)
    test_perplexity = calculate_perplexity(ngram_model, test_corpus)

    print(f"Training Perplexity: {train_perplexity}")
    print(f"Test Perplexity: {test_perplexity}")
    
    # Write generated code to CSV
    output(ngram_model, test_corpus, output_file_path, num_samples=num_samples)

    return ngram_model
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        csv_file_path = 'java.csv'
        column_name = 'code'  # Column in CSV containing Java methods
        main(csv_file_path, column_name, n=3, test_size=0.2, output_file_path='generated_codes.csv', num_samples=100, k=0.01)
    else:
        try:
            inputs = [sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5], int(sys.argv[6]), float(sys.argv[7])]
        except:
            sys.exit("Usage: python ngram.py\nor\npython ngram.py data_path method_column n test_size output_path output_samples k")

        if not ("csv" in inputs[0].split(".") and os.path.exists(inputs[0])):
            sys.exit("Invalid data path.\nUsage: python ngram.py\nor\npython ngram.py data_path method_column n test_size output_path output_samples k")
        elif not inputs[1].isalpha():
            sys.exit("Invalid column name.\nUsage: python ngram.py\nor\npython ngram.py data_path method_column n test_size output_path output_samples k")
        elif not ("csv" in inputs[4].split(".")):
            sys.exit("Invalid output path.\nUsage: python ngram.py\nor\npython ngram.py data_path method_column n test_size output_path output_samples k")

        main(inputs[0], inputs[1], n=inputs[2], test_size=inputs[3], output_file_path=inputs[4], num_samples=inputs[5], k=inputs[6])

