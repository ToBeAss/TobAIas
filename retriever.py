class Retriever:
    def __init__(self, search_type: str, k: int, score_threshold: float):
        self._search_type = search_type
        self._k = k
        self._score_threshold = score_threshold
    
    