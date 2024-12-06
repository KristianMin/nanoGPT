import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import glob

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

model = "4o"
qa_r = 0.2

if __name__ == '__main__':
    dataset = load_dataset("KrisMinchev/LLMCollapseBios", f"data_{model}", num_proc=num_proc_load_dataset)
    dataset = dataset.shuffle(100000)
    qa = load_dataset("KrisMinchev/LLMCollapseBios", f"qa_{model}", num_proc=num_proc_load_dataset)
    qa = qa.shuffle(100000)
    split_qa = qa["train"].train_test_split(test_size=qa_r, shuffle=True)
    split_qa["val"] = split_qa.pop("test")


    def process_bio(example):
        ids = enc.encode_ordinary(example['output'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized_bio = dataset.map(
        process_bio,
        remove_columns=['output'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print(tokenized_bio)

    def process_qa(example):
        ids = enc.encode_ordinary(example['question'] + example['answer'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized_qa = split_qa.map(
        process_qa,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    for split, dset in tokenized_bio.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'./{split}_bio.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    for split, dset in tokenized_qa.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'./{split}_qa.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


    files = [os.path.join(os.path.dirname(__file__), 'train_bio.bin'),
             os.path.join(os.path.dirname(__file__), 'train_qa.bin')]
    out_data = b''
    for fn in files:
        with open(fn, 'rb') as fp:
            out_data += fp.read()
    with open(os.path.join(os.path.dirname(__file__), 'train.bin'), 'wb') as fp:
        fp.write(out_data)

    os.rename(os.path.join(os.path.dirname(__file__), 'val_qa.bin'), os.path.join(os.path.dirname(__file__), 'val.bin'))

    for f in glob.glob(os.path.join(os.path.dirname(__file__), 'train_*.bin')):
        os.remove(f)
