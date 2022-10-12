## Training data examples and validation data examples
* training_src.txt: 
the source code training data, each line is an instance in the following format:
buggy line &lt;CTX&gt; surrounding function \t fixed line
* training_tokenize.txt:
the training data tokenized by spaces, strings, numbers, camel letters and udnerscores.
* training_bpe.txt:
the training data tokenized by subword-tokenizer
* prepare_training_data.py:
the code for preparing/preprocessing training data

## Prepare Test Input
To prepare test input for new test data, 
* first run the `prepare_cure_input` function in `prepare_test_input.py`, argument `buggy_file` is the path to the file that contains the buggy code.
* then run commands:
```
subword-nmt apply-bpe -c ../vocabulary/subword.txt < input.txt > input_bpe.txt
subword-nmt apply-bpe -c ../vocabulary/subword.txt < identifier.tokens > identifier_bpe.tokens
```
* thirdly, comment the `prepare_cure_input` function and uncomment the `clean_testing_bpe`function to run. This function finalize the input files
* the final `input_bpe.txt`, `identifier.txt` and `identifier_bpe.tokens` are the files passed to the src/tester/generator.py.

You can write your own script to call the `prepare_cure_input` function, `subword-nmt` command, and call the `clean_testing_bpe` function sequentially.
