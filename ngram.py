import sys
import os
import pandas as pd
import pickle
from collections import defaultdict, Counter
import random
import math
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
                self.vocab.add(token)
            for i in range(len(padded_tokens) - self.n + 1):
                context = tuple(padded_tokens[i:i+self.n-1])
                token = padded_tokens[i+self.n-1]
                self.ngrams[context][token] += 1
                self.context_totals[context] += 1
        self.vocab_size = len(self.vocab)

    def get_prob(self, context, token):
        context = tuple(context)
        token_count = self.ngrams[context][token] if context in self.ngrams else 0
        context_count = self.context_totals[context] if context in self.context_totals else 0
        return (token_count + self.k) / (context_count + self.k * self.vocab_size)
    
    def unk_prob(self, context):
        return self.get_prob(context, '<unk>')

    def generate_code(self, context, max_length):
        context = ['<s>'] * (self.n - 1) + context
        generated = []

        for _ in range(max_length):
            context_tuple = tuple(context[-(self.n-1):])

            if context_tuple not in self.ngrams:
                next_token = '<unk>'
            else:
                tokens, counts = zip(*self.ngrams[context_tuple].items())
                prob_dist = [self.get_prob(context_tuple, token) for token in tokens]
                tokens = list(tokens) + ['<unk>']
                prob_dist = list(prob_dist) + [self.unk_prob(context_tuple)]
                next_token = random.choices(tokens, weights=prob_dist, k=1)[0]
            
            if next_token == '</s>':
                break
            
            generated.append(next_token)
            context.append(next_token)

        return generated


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


def evaluate_accuracy(model, test_corpus):
    total_tokens = 0
    correct_tokens = 0
    
    for test_tokens in test_corpus:
        context = test_tokens[:model.n-1]  # Use the first n-1 tokens as context
        max_length = len(test_tokens)  # Set max_length to the length of the current test function
        generated_tokens = model.generate_code(context, max_length=max_length)
        
        # Compare generated tokens with the actual test tokens
        for generated_token, original_token in zip(generated_tokens, test_tokens):
            total_tokens += 1
            if generated_token in test_tokens:
                correct_tokens += 1
                
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return accuracy


def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_data(file_path, column_name):
    """ Loads Java methods from a CSV file """
    df = pd.read_csv(file_path)
    methods = df[column_name].dropna()
    corpus = [preprocess_code(method) for method in methods]
    return corpus


def output(model, test_corpus, output_file_path):
    """ Writes generated code from the N-gram model to a CSV file without probabilities """
    results = []
    for test_tokens in test_corpus:
        context = test_tokens[:model.n-1]  # Use the first n-1 tokens as the context
        max_length = len(test_tokens)  # Set max_length to the length of the current test function
        generated_code = model.generate_code(context, max_length=max_length)
        results.append({
            'input_context': ' '.join(test_tokens),
            'generated_code': ' '.join(generated_code)
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file_path, index=False)


def main(test_file, column_name='code', n=2, model_file='ngram_model.pkl', train_file='train.csv', k=0.01):
    # Check if the model file exists
    if os.path.exists(model_file):
        print("Loading the existing model...")
        ngram_model = load_model(model_file)
    else:
        print("Training a new model...")
        # Load the training data
        train_corpus = load_data(train_file, column_name)
        
        # Initialize and train the N-gram model
        ngram_model = NgramModel(n, k)
        ngram_model.train(train_corpus)
        
        # Save the trained model
        save_model(ngram_model, model_file)
    
    # Load the test data
    test_corpus = load_data(test_file, column_name)
    test_100 = test_corpus[:100]
    
    # Evaluate accuracy on the test data
    test_perplexity = calculate_perplexity(ngram_model, test_corpus)
    print(f"Test Perplexity: {test_perplexity}")

    #accuracy = evaluate_accuracy(ngram_model, test_corpus)
    accuracy = evaluate_accuracy(ngram_model, test_100)
    print(f"Test Token Match Accuracy: {accuracy * 100:.2f}%")
    
    # Write generated code to CSV
    output(ngram_model, test_100, 'generated_codes.csv')
    
    return ngram_model


if __name__ == "__main__":
    if len(sys.argv) == 2:
        test_file = sys.argv[1]
        main(test_file)
    else:
        print("Usage: python ngram.py <test_file>")
