#Instructions

For this program, both training and testing data should be tokenized in a CSV file (spreadsheet), as described here:
https://docs.cortext.net/data-formats/#:~:text=by%20CorText%20Manager.-,Csv,December%2023rd%202001%20for%20instance)
Name the file containing your training data train.csv. The file containing your test data can be named whatever you like. We recommend test.csv.

To run the program, open a terminal of your choice and navigate to the directory containing all of the files. Then run the command:

python ngram.py test.csv

Where test.csv is replaced with the name of your test data file, including the file extension.


#Fields

code: a method used for the model
code_tokens: the above method, parsed into an array of tokens
docstring: any documentation associated with the above method
docstring_tokens: the above docstring, parsed into an array of tokens
func_name: name of the method
language: the language the method is written in
original_string: the original method before any parsing processes
partition: a token or string used to indicate a break between methods
path: the path to the original file the method was pulled from
repo: the repository the method was pulled from
sha:
url: the URL to the original file within its original repository
