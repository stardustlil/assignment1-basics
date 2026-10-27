from cs336_basics.train_bpe import merge_word, Pair
import regex as re
from cs336_basics.pre_tokenizer import PAT
from collections.abc import Iterable,Iterator
import pickle

class Tokenizer():
    def __init__(self, vocab:dict[int,bytes], merges:list[Pair], special_tokens:list[str] | None= None):
        self.vocab = dict(vocab)
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [], key = len, reverse=True)
        existing = set(self.vocab.values())
        
        for st in self.special_tokens:
            b = st.encode('utf-8')
            if b not in existing:
                self.vocab[len(self.vocab)] = b
                existing.add(b)
        
        self.byte_to_id = {b:i for i,b in self.vocab.items()}
        self.merge_rank = {pair:i for i , pair in enumerate(merges)}
        #special pattern
        self.special_set = set(self.special_tokens)
        if self.special_tokens:
            escaped = [re.escape(t) for t in self.special_tokens]
            self.special_patterns = "(" + "|".join(escaped) + ")"
        else:
            self.special_patterns = None

    @classmethod
    def from_files(cls, vocab_path:str, merges_path:str, special_tokens:list[str] | None = None) ->"Tokenizer":
        with open(vocab_path , "rb") as f:
            vocab = pickle.load(f)
        with open(merges_path , "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges , special_tokens)


    def _encode_pretoken(self, s: str) -> list[int]:
        word = tuple(bytes([x]) for x in s.encode('utf-8'))
        while True:
            best_pair = None
            best_rank = float('inf')
            for pair in zip(word[:-1] , word[1:]):
                rank = self.merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank
            if best_pair is None:
                break    
            word = merge_word(word, best_pair)
        
        return [self.byte_to_id[b] for b in word]

    def encode(self, text:str) -> list[int]:
        #保留sptonkens并切分
        if self.special_patterns:
            parts = re.split(self.special_patterns, text)
        else:
            parts = [text]
        result = []
        for s in parts:
            if s in self.special_set:
                result.append(self.byte_to_id[s.encode('utf-8')])
            else:
                for m in re.finditer(PAT, s):
                    result.extend(self._encode_pretoken(m.group()))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids:list[int]) -> str:
        bytess = b"".join(self.vocab[id] for id in ids)
        return bytess.decode('utf-8' , errors='replace')
        



        
    
        
               