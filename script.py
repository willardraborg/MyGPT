import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available else 'cpu'
eval_iters = 200
#--------------------------


with open('KJV-Bible.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

#get unique characters in text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

#tokenize characters (char->int)
str_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_str = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [str_to_int[c] for c in s]
def decode(l):
    return ''.join([int_to_str[i] for i in l])

print(encode("test"))
print(decode(encode("test")))

# encode the entire text and store in Tensor
import torch
data = torch.tensor(encode(text), dtype = torch.long)
print(data.shape, data.dtype)
print(data[:1000])

# split data into training and validation sets
n = int(0.9 * len(data))
train = data[:n]
test = data[n:]

block_size = 8
train[:block_size+1]

x = train[:block_size]
y = train[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")

batch_size = 4 # number of independent sequences to process in parallel
block_size = 8 # maximum context length

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # get x block
    x = torch.stack([data[i:i+block_size] for i in ix])
    # get y block (1 offset from x)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('-----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        # x and y batches are already offset so we dont offset them here
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C). Each index replaced by row of scores

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :] # (B,T,C) -> (B,C), last element in time dimension
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # sample from probabilities (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(10000):

    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(loss.item())

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=300)[0].tolist()))