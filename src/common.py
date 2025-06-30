import os
import random
from datetime import datetime
from tqdm import tqdm
import subprocess


def get_file_line_count(file_path):
    line_count = 0
    buffer_size = 1024 * 1024
    with open(file_path, 'r') as file:
        buffer = file.read(buffer_size)
        while buffer:
            line_count += buffer.count('\n')
            buffer = file.read(buffer_size)
    return line_count


def create_batch_from_iterator(iterator, processor, batch_size):
    batch = []
    for item in iterator:
        x = processor(item)
        if x is None:
            continue
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_temp_file(ext=''):
    directory = os.environ.get("CACHE_DIR", "/tmp/")
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    file_name = datetime.now().strftime("%Y%m%d_%H:%M:%S_") + str(random.randint(100000000, 999999999)) + ext
    return os.path.join(directory, file_name)


def count_lines_using_subprocess(file_path):
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE)
    num_lines = int(result.stdout.split()[0])
    return num_lines


def get_dataset(table_path, col_names="", mute=True):
    col_list = col_names.split(',')
    reader = open(table_path, 'r')
    row_count = count_lines_using_subprocess(table_path)
    if not mute:
        tqbar = tqdm(desc="Read data from odps", total=row_count)
    read_count = 0
    results = []
    while True:
        x = reader.readline()
        if x == '':
            break
        if read_count == 0:
            assert len(col_list) == len(x.split(','))
        vals = x.split(',')
        results.append(dict(zip(col_list, vals)))
        if not mute:
            tqbar.update(1)
        read_count += 1
    reader.close()
    return results


def latest_checkpoint(model_dir):
    max_step = -1
    for fname in os.listdir(model_dir):
        if fname.startswith('checkpoint-'):
            step = int(fname.replace('checkpoint-', '').replace('.pth', ''))
            max_step = max(max_step, step)
    if max_step > -1:
        return os.path.join(model_dir, f"checkpoint-{max_step}.pth")
