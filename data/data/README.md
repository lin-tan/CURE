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