import bm25s
import Stemmer
from fastapi import FastAPI

stemmer = Stemmer.Stemmer("english")
retriever = bm25s.BM25.load("index", load_corpus=True)

app = FastAPI()

@app.get("/query")
async def query(q: str, k : int = 10):
    query_tokens = bm25s.tokenize(q, stemmer=stemmer)
    results = retriever.retrieve(query_tokens, k=k)
    formatted_results = list(map(lambda doc, score: {
        "text": doc['text'],
        "score": float(score)
    }, results.documents[0], results.scores[0]))
    return formatted_results

@app.post("/index", status_code=201)
async def index(add_corpus: list[str]):
    old_corpus = [item['text'] for item in retriever.corpus]
    new_corpus = list(old_corpus) + add_corpus
    corpus_tokens = bm25s.tokenize(new_corpus, stopwords="en", stemmer=stemmer)
    retriever.index(corpus_tokens)
    retriever.save("index", corpus=new_corpus)
    return 'ok'
