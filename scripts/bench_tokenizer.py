from cs336_basics import tokenizer
import time
import numpy as np
import resource

def bench_mark(bts:bytes , tokenizer):
        byte_len = len(bts)
        text = bts.decode('utf-8')
        text_len = len(text)
        t = time.time()
        encode_output = tokenizer.encode(text)
        cost_time = time.time() - t
        print(f"耗时: {cost_time:.2f} 秒")
        compress_ratio = byte_len / len(encode_output)
        throughput = byte_len / cost_time / 1e6
        print(f"压缩率 {compress_ratio:.2f}")
        print(f"thoughput {throughput:.2f} MB/s")
        return encode_output

def chunk_encode(tokenizer:tokenizer.Tokenizer, in_path:str , out_path:str):
    with open(in_path, "r" , encoding="utf-8") as f:
        t = time.time()
        ids = np.fromiter(tokenizer.encode_iterable(f) , dtype=np.uint16)
        print(f"耗时: {time.time() - t:.2f} 秒")
        #主进程峰值内存
        main_mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
        print(f"主进程峰值内存 {main_mem:.2f} GB")


    np.save(out_path, ids)
    return ids

    
if __name__ == "__main__":
    import time
    tok_off = tokenizer.Tokenizer.from_files("output/ts_vocab.pkl" , "output/ts_merges.pkl" , ["<|endoftext|>"])
    tok_on  = tokenizer.Tokenizer.from_files("output/ts_vocab.pkl" , "output/ts_merges.pkl" , ["<|endoftext|>"] , 200_000)
    # with open("data/TinyStoriesV2-GPT4-valid.txt" , "rb") as f:
    #     bts = f.read()
    # print("不开缓存")
    # r1 = bench_mark(bts, tok_off)
    # print("开缓存")
    # r2 = bench_mark(bts, tok_on)
    # print(f"一致性检查 {r1 == r2}")
    
    ids_train = chunk_encode(tok_on, "data/TinyStoriesV2-GPT4-train.txt" , "output/train_encode")
    print(len(ids_train))