"""
http://ethen8181.github.io/machine-learning/search/bm25_intro.html
"""

import MLJ:fitted_params
using MLJ
import TextAnalysis

BM25Transformer = @load BM25Transformer pkg=MLJText

docs = ["Hi my name is Sam.", "How are you today?"]
bm25_transformer = BM25Transformer()

tokenized_docs = TextAnalysis.tokenize.(docs)

mach = machine(bm25_transformer, tokenized_docs)|>fit!

fitted_params(mach)