from collections import Counter
from cs336_basics import pre_tokenizer
import os
import heapq



Word = tuple[bytes, ...]
Pair = tuple[bytes, bytes]

def get_pair_counts(word_counts:dict[Word, int]) -> dict[Pair, int]:
    count = Counter()
    for w, c in word_counts.items():
        for a , b in zip(w[:-1] , w[1:]):
            count[(a,b)] += c
    return count


def select_best_pair(pair_counts: dict[Pair, int]) -> Pair:
    return max(pair_counts, key=lambda p: (pair_counts[p], p))

def merge_word(word:Word , pair:Pair) -> Word:
    merged = pair[0] + pair[1]
    result = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and pair[0] == word[i] and pair[1] == word[i+1]:
            result.append(merged)
            i += 2
        else:
            result.append(word[i])
            i += 1
    return tuple(result)

def apply_merge(word_counts:dict[Word,int] , pair:Pair) -> dict[Word, int]:
    new_counts = {}
    for word, freq in word_counts.items():
        new_counts[merge_word(word, pair)] = freq
    return new_counts
    

def run_merges(word_counts:dict[Word, int], num_merges:int) -> list[Pair]:
    merges = []
    for i in range(num_merges):   
        pair_counts = get_pair_counts(word_counts)
        if not pair_counts:
            break
        best_pair   = select_best_pair(pair_counts)
        merges.append(best_pair)
        word_counts = apply_merge(word_counts, best_pair)
    return merges


def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[Pair]]: # (vocab, merges)
    word_counts = pre_tokenizer.pretokenize(input_path, special_tokens)
    num_merges = vocab_size - 256 - len(special_tokens)
    # init vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for st in special_tokens:
        vocab[len(vocab)] = st.encode('utf-8')
    merges = run_merges(word_counts, num_merges)
    for a,b in merges:
        vocab[len(vocab)] = a+b
    return vocab, merges

class RevPair:
    __slots__ = ("pair",)              # 只是省内存的写法，可以先不管
    def __init__(self, pair):
        self.pair = pair
    def __lt__(self, other):           # 定义 a < b 的含义
        return self.pair > other.pair  # 反过来：pair 越大，越「小」


def build_heap(pair_counts: dict[Pair, int]) -> list:
    heap = [(-c, RevPair(p)) for p, c in pair_counts.items()]
    heapq.heapify(heap)        # 把一个普通 list 一次性整理成堆，比逐个 push 快
    return heap

def pop_best_pair(heap: list, pair_counts: dict[Pair, int]) -> Pair | None:
    while heap:
        neg_c, rp = heapq.heappop(heap)
        if pair_counts.get(rp.pair) == -neg_c:
            return rp.pair
    return None


def build_index(word_counts: dict[Word, int]) -> dict[Pair, set[Word]]:
    result = {}
    for w in word_counts:
        for a, b in zip(w[:-1] , w[1:]):
            result.setdefault((a,b), set()).add(w)
    return result

def _bump(pair_counts: dict[Pair, int], pair: Pair, delta: int , change:set[Pair]) -> None:
    v = pair_counts.get(pair, 0) + delta
    if v > 0:
        pair_counts[pair] = v
    else:
        pair_counts.pop(pair, None)
    change.add(pair)

def run_merge_fast(word_counts:dict[Word, int], num_merges:int) -> list[Pair]:
    word_counts = dict(word_counts) #拷贝一次
    pair_counts = get_pair_counts(word_counts)
    index = build_index(word_counts)
    heap = build_heap(pair_counts)
    merges = []
    
    for i in range(num_merges):
        best_pair = pop_best_pair(heap, pair_counts)
        if best_pair is None:
            break

        changes = set() #记录有变化的 pairs
        words = list(index[best_pair])
        new_merge = best_pair[0] + best_pair[1]
        merges.append((best_pair[0],best_pair[1]))
        for w in words:
            #减去贡献 and 增加贡献
            freq = word_counts[w]
            i = 0
            new_w = []
            while i < len(w):
                if i < len(w) -1 and w[i]==best_pair[0] and w[i+1] == best_pair[1]:
                    #自身减
                    _bump(pair_counts, best_pair, -freq , changes)
                    if i < len(w) - 2:
                        #左边-
                        _bump(pair_counts, (w[i+1],w[i+2]), -freq, changes)
                        #左边+
                        _bump(pair_counts, (new_merge, w[i+2]) , freq, changes)
                    if i > 0:
                        #右边-
                        _bump(pair_counts, (new_w[-1], w[i]), -freq, changes)
                        #右边+
                        _bump(pair_counts, (new_w[-1], new_merge), freq, changes)
                    i += 2
                    new_w.append(new_merge)
                else:
                    new_w.append(w[i])
                    i += 1
            new_w = tuple(new_w)
            # 维护 word_counts
            del word_counts[w]
            word_counts[new_w] = freq
            # 维护index
            for a , b in zip(w[:-1], w[1:]):
                index[(a,b)].discard(w)
            for a , b in zip(new_w[:-1], new_w[1:]):
                index.setdefault((a,b),set()).add(new_w)
        # push有变化的pair
        for p in changes:
            if p in pair_counts:
                heapq.heappush(heap, (-pair_counts[p] , RevPair(p)))
        
    return merges


def train_bpe2(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[Pair]]: # (vocab, merges)
    word_counts = pre_tokenizer.pretokenize(input_path, special_tokens)
    num_merges = vocab_size - 256 - len(special_tokens)
    # init vocab
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for st in special_tokens:
        vocab[len(vocab)] = st.encode('utf-8')    
    
    merges = run_merge_fast(word_counts, num_merges)
    
    for a,b in merges:
        vocab[len(vocab)] = a+b
    return vocab, merges
            

if __name__ == "__main__":
    import pickle, time, resource

    t = time.time()
    vocab, merges = train_bpe2("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    elapsed = time.time() - t

    # 内存峰值（macOS 上单位是 byte）
    main_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    child_mem = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**3

    print(f"总耗时：{elapsed:.1f} 秒")
    print(f"主进程内存峰值：{main_mem:.2f} GB")
    print(f"单个子进程内存峰值：{child_mem:.2f} GB")
    print(f"vocab 大小：{len(vocab)}，merge 次数：{len(merges)}")

    with open("ts_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("ts_merges.pkl", "wb") as f:
        pickle.dump(merges, f)