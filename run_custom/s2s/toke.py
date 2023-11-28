
__all__ = [
    'IntVocab', 'FloatVocab', 'StrVocab',
]
class IntVocab:
    r"""
    maps symbols to tokens
        symbols can be any datatype <--- will be converted to int
        tokens are tt.long or integer type
    """

    # special symbols
    UKN, BOS, EOS, PAD  = -1, -2, -3, -4

    def __init__(self, symbols) -> None:
        self.vocab = {} # value v/s symbol
        self.vocab[self.UKN] =  0
        self.vocab[self.BOS] =  1
        self.vocab[self.EOS] =  2
        self.vocab[self.PAD] =  3
        for i,symbol in enumerate(symbols): self.vocab[int(symbol)] = i + 4 #<-- offset
        self.rvocab = list(self.vocab.keys())
        self.count = len(self.rvocab)

    def __len__(self): return self.count

    
    # inplcae forward
    def forward_(self, symbols, dest):
        for i,symbol in enumerate(symbols): dest[i] = self.vocab[int(symbol)]
    def forward1_(self, symbol, dest, i): dest[i] = self.vocab[int(symbol)]
    
    # forward converts symbol to token
    def forward(self, symbols): return [self.vocab[int(symbol)] for symbol in symbols]
    def forward1(self, symbol):  return self.vocab[int(symbol)]

    # backward converts token to symbol
    def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
    def backward1(self, token): return self.rvocab[int(token)]

class FloatVocab:
    r"""
    maps symbols to tokens
        symbols can be any datatype <--- will be converted to float
        tokens are tt.long or integer type
    """

    # special symbols
    UKN, BOS, EOS, PAD  = -1., -2., -3., -4.

    def __init__(self, symbols) -> None:
        self.vocab = {} # value v/s symbol
        self.vocab[self.UKN] =  0
        self.vocab[self.BOS] =  1
        self.vocab[self.EOS] =  2
        self.vocab[self.PAD] =  3
        for i,symbol in enumerate(symbols): self.vocab[float(symbol)] = i + 4 #<-- offset
        self.rvocab = list(self.vocab.keys())
        self.count = len(self.rvocab)

    def __len__(self): return self.count

    
    # inplcae forward
    def forward_(self, symbols, dest):
        for i,symbol in enumerate(symbols): dest[i] = self.vocab[float(symbol)]
    def forward1_(self, symbol, dest, i): dest[i] = self.vocab[float(symbol)]
    
    # forward converts symbol to token
    def forward(self, symbols): return [self.vocab[float(symbol)] for symbol in symbols]
    def forward1(self, symbol):  return self.vocab[float(symbol)]

    # backward converts token to symbol
    def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
    def backward1(self, token): return self.rvocab[int(token)]

class StrVocab:
    r"""
    maps symbols to tokens
        symbols can be any datatype <--- will be converted to str
        tokens are tt.long or integer type
    """

    # special symbols
    UKN, BOS, EOS, PAD  = "<UKN>", "<BOS>", "<EOS>", "<PAD>"

    def __init__(self, symbols) -> None:
        self.vocab = {} # value v/s symbol
        self.vocab[self.UKN] =  0
        self.vocab[self.BOS] =  1
        self.vocab[self.EOS] =  2
        self.vocab[self.PAD] =  3
        
        for i,symbol in enumerate( symbols ): self.vocab[symbol] = i + 4 #<-- offset
        self.rvocab = list(self.vocab.keys())
        self.count = len(self.rvocab)

    def __len__(self): return self.count

    
    # inplcae forward
    def forward_(self, symbols, dest):
        for i,symbol in enumerate(symbols): dest[i] = self.vocab[symbol]
    def forward1_(self, symbol, dest, i): dest[i] = self.vocab[symbol]
    
    # forward converts symbol to token
    def forward(self, symbols): return [self.vocab[symbol] for symbol in symbols]
    def forward1(self, symbol):  return self.vocab[symbol]

    # backward converts token to symbol
    def backward(self, tokens): return [self.rvocab[int(token)] for token in tokens]
    def backward1(self, token): return self.rvocab[int(token)]

