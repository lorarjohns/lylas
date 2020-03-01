import numpy as np
import re
from scipy.sparse import csr_matrix 

class StringMatcher:
    def __init__(self):
        pass
    def ngrams(self, string, n=3):
        # string = re.sub(r"", r"", string)
        ngrams = zip(*[string[i:] for i in range(n)])
        return ["".join(ngram) for ngram in ngrams]
    
    def get_matches_df(self, sparse_matrix)

        import sparse_dot_topn.sparse_dot_topn as ct