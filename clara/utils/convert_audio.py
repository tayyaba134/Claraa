import os
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

SRC_FILE_EXT = '.m4a'
DEST_FILE_EXT = '.flac'

def audio_to_flac(audio_in_path, audio_out_path, sample_rate=48000, no_log=True, segment_start: float = 0, segment_end: float = None):
    log_cmd = ' -v quiet' if no_log else ''
    segment_cmd = f'-ss {segment_start} -to {segment_end}' if segment_end else ''
    cmd = (
        f'ffmpeg -y -i "{audio_in_path}" -vn {log_cmd} -flags +bitexact '
        f'-ar {sample_rate} -ac 1 {segment_cmd} "{audio_out_path}"'
    )
    if os.system(cmd) != 0:
        print(f"Error processing {audio_in_path}")

def create_threads(src_paths, dest_root_path, max_workers=96):
    if not os.path.exists(dest_root_path):
        os.makedirs(dest_root_path, exist_ok=True)

    with tqdm.tqdm(total=len(src_paths), desc=f'Processing {dest_root_path}') as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(audio_to_flac, path, 
                                os.path.join(dest_root_path, os.path.basename(path).replace(SRC_FILE_EXT, DEST_FILE_EXT))): 
                path for path in src_paths
            }

            for future in as_completed(futures):
                pbar.update(1)
                if future.exception() is not None:
                    print(f"Error processing file: {futures[future]}, {future.exception()}")

if __name__ == '__main__':
    import glob
    src_paths = glob.glob('/fsx/knoriy/tmp/VoxCeleb_gender/**/*.m4a', recursive=True)
    dest_root_path = '/fsx/knoriy/tmp/flac_voxceleb/'

    create_threads(src_paths, dest_root_path, max_workers=96)
