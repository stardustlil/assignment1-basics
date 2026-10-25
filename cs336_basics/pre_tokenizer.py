import regex as re
import os
from collections import Counter
from typing import BinaryIO
from multiprocessing import Pool 

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_on_sptokens(text, special_tokens) -> list:
    if not special_tokens:
        return [text]
    sptokens = sorted(special_tokens , key=len, reverse=True)
    escaped = []
    for t in sptokens:
        escaped.append(re.escape(t))
    pattern = '|'.join(escaped)
    return re.split(pattern, text)


def count_pretokens(text, special_tokens) -> dict:
    count = Counter()
    segs = split_on_sptokens(text, special_tokens)
    for seg in segs:
        count.update(m.group() for m in re.finditer(PAT, seg))
    return count

def _count_chunk(args) -> dict:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8")
    return count_pretokens(text , special_tokens)



def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def pretokenize(input_path, special_tokens, num_processes = None, chunk_bytes = 64 * 1024 * 1024 , split_token = "<|endoftext|>") -> dict:
    if num_processes is None:
        num_processes = os.cpu_count() or 1

    chunk_nums = max(num_processes, os.path.getsize(input_path) // chunk_bytes)
    
    with open(input_path , "rb") as f:
        boundaries = find_chunk_boundaries(f, chunk_nums, split_token.encode('utf-8'))
    jobs = []

    for start, end in zip(boundaries[:-1] , boundaries[1:]):
        jobs.append((input_path, start , end , special_tokens))

    tot = Counter()
    with Pool(num_processes) as pool:
        for d in pool.imap_unordered(_count_chunk , jobs):
            tot.update(d)
            del d
        
    result_counts = {}
    while tot:
        s , c = tot.popitem()
        parts = []
        for x in s.encode('utf-8'):
            parts.append(bytes([x]))
        result_counts[tuple(parts)] = c
    return result_counts





if __name__ == "__main__":
    import pickle, time

    cache = "ts_train_pretok.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            wc = pickle.load(f)
        print("read on cache")
    else:
        t = time.time()
        wc = pretokenize("data/TinyStoriesV2-GPT4-train.txt", ["<|endoftext|>"])
        print(f"耗时{time.time() - t:.1f}秒")
        with open(cache , "wb") as f:
            pickle.dump(wc, f)

    print(f"不重复的 pre-token 数：{len(wc)}")
    print(f"pre-token 总数：{sum(wc.values())}")
