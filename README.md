# Walkthrough

Starting off we extract all unique characters from the text to get a vocabulary. We then tokenize our vocabulary so that each unique character has a unique integer representation.

After tokenizing the data it is split into a training and a test set (90/10 split).

The tokenized data gets processed in chunks of size `block_size`

```python
block_size = 8
train[:block_size+1]
```
The reason `train` is of size `block_size+1` is because of the following procedure

```python
x = train[:block_size]
y = train[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
```

For each number of characters in a block (1-8) we compute its target. For example, for a 3 character context we could get