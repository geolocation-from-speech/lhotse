"""
Audioset is a large-scale dataset of manually annotated audio events.
This recipe is for the temporally-strong labels released in 2021 and available from https://research.google.com/audioset/download_strong.html.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from tqdm.auto import tqdm

from lhotse import CutSet, fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, is_module_available, resumable_download

class loggerSilenceErrors:
    def error(msg):
        pass

    def warning(msg):
        pass

    def debug(msg):
        pass

def download_audioset(
    target_dir: Pathlike = ".",
    num_jobs: int = 1,
) -> Path:
    """
    Download the Audioset dataset.
    :param target_dir: Pathlike, the path of the dir to store the dataset.
    :param num_jobs: int, number of parallel threads used for 'download_single' calls.
    :return: the path to downloaded data and the JSON file.
    """
    if is_module_available("yt_dlp"):
        import yt_dlp
    else:
        raise ImportError(
            "To download the audioset dataset, please install the optional dependency: pip install yt-dlp"
        )
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = target_dir / "audioset"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Download tsv files
    resumable_download(
    "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset_train_strong.tsv",
    filename=corpus_dir / "audioset_train_strong.tsv",
    )

    resumable_download(
    "http://storage.googleapis.com/us_audioset/youtube_corpus/strong/audioset_eval_strong.tsv",
    filename=corpus_dir / "audioset_eval_strong.tsv",
    )

    dataset_parts = ("train", "eval")

    

    for part in dataset_parts:
        downloads = defaultdict(set)
        with open(corpus_dir / f"audioset_{part}_strong.tsv") as f:
            for annotation in f.readlines()[1:]:
                segment_id, _ = annotation.split("\t", 1)
                youtube_id, start_time = segment_id.rsplit("_", 1)
                start_time = int(start_time) // 1000
                downloads[start_time].add(youtube_id)
            output_dir = corpus_dir / part
            for start_time, youtube_ids in tqdm(downloads.items(), desc=f"Downloading Audioset {part}"):
                ydl_opts = {
                    "format": "wav/bestaudio/best",
                    "outtmpl": f"{output_dir}/%(id)s",
                    "download_ranges": yt_dlp.utils.download_range_func(
                        [], [[start_time, start_time + 10.0]]
                    ),
                    "force_keyframes_at_cuts": True,
                    "quiet": True,
                    "ignoreerrors": True,
                    "logger": loggerSilenceErrors,
                    "postprocessor_args": {"ffmpeg": ["-ar", "16000"]},
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "wav",
                        }
                    ],
                }
                download_links = [f"https://youtube.com/watch?v={youtube_id}" for youtube_id in youtube_ids]
                for _ in tqdm(parallel_map(download_segment, download_links, (ydl_opts for i in download_links), num_jobs=num_jobs), desc=f"Downloading segment", total=len(download_links)):
                    pass

    return corpus_dir


def download_segment(
    link: str,
    ydl_opts: dict,
) -> Path:
    """
    Downloads a single 10 second audio clip from a given Youtube ID.
    :param link: a tuple containing the Youtube ID and the start time of the Youtube video to download from
    :param ydl_opts: Pathlike, to which directory to download to.
    :return: the path to which the audio was downloaded to.
    """
    import yt_dlp

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])


def prepare_audioset(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike],
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param num_jobs: int, number of parallel threads used for 'parse_annotation' calls.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    parts = ("train", "eval")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Maybe some manifests already exist: we can read them and save a bit of preparation time.
    manifests = read_manifests_if_cached(
        dataset_parts=parts,
        output_dir=output_dir,
        prefix="audioset",
        suffix="jsonl.gz",
        lazy=True,
    )

    for part in parts:
        logging.info(f"Processing Audioset subset: {part}")
        if manifests_exist(
            part=part, output_dir=output_dir, prefix="audioset", suffix="jsonl.gz"
        ):
            logging.info(f"Audioset subset: {part} already prepared - skipping.")
            continue

        filename = corpus_dir / f"audioset_{part}_strong.tsv"
        # Remove duplicate annotations
        with open(filename, "r", encoding="utf-8") as f:
            items = list(set(f.readlines()[1:]))

        lookup = {}
        mid_to_display_name = corpus_dir / "mid_to_display_name.tsv"
        with open(mid_to_display_name) as f:
            for i in f.readlines():
                mid, display_name = i.strip().split("\t")
                lookup[mid] = f"[{display_name}]"

        items_dict = defaultdict(list)

        for annotation in tqdm(items, desc=f"Initializing {part} split"):
            segment_id, _ = annotation.split("\t", 1)
            segment_id, _ = segment_id.rsplit("_", 1)
            file_path = corpus_dir / part / f"{segment_id}.wav"
            if file_path.exists():
                items_dict[segment_id].append(annotation)
        
        with RecordingSet.open_writer(
            output_dir / f"audioset_recordings_{part}.jsonl.gz"
        ) as rec_writer, SupervisionSet.open_writer(
            output_dir / f"audioset_supervisions_{part}.jsonl.gz"
        ) as sup_writer:
            data_dir = corpus_dir / part
            for recording, segments in tqdm(
                parallel_map(
                    parse_annotation,
                    items_dict.values(),
                    (data_dir for i in items),
                    (lookup for i in items),
                    num_jobs=num_jobs,
                ),
                desc=f"Processing Audioset {part} entries",
                total=len(items_dict.values()),
            ):
                if recording is None:
                    continue
                # Filter warnings
                logging.getLogger().setLevel(logging.ERROR)

                # Fix and validate the recording + supervisions
                recordings, segments = fix_manifests(
                    recordings=RecordingSet.from_recordings([recording]),
                    supervisions=SupervisionSet.from_segments(segments),
                )
                
                validate_recordings_and_supervisions(
                    recordings=recordings, supervisions=segments
                )
                logging.getLogger().setLevel(logging.WARNING)

                # Write the manifests
                rec_writer.write(recordings[0])
                for segment in segments:
                    sup_writer.write(segment)

        manifests[part] = {
            "recordings": RecordingSet.from_jsonl_lazy(rec_writer.path),
            "supervisions": SupervisionSet.from_jsonl_lazy(sup_writer.path),
        }

    return dict(manifests)

def parse_annotation(
    item: Any, corpus_dir: Pathlike, lookup: Dict,
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    """
    Process a single annotation from the Audioset dataset.
    :param item: The annotation to process.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param lookup: Dict, a lookup table of MIDs and display names.
    :return: A tuple containing the Recording and SupervisionSegment.
    """

    segment_id, _ = item[0].strip().split("\t", 1)
    youtube_id = segment_id.rsplit("_", 1)[0]
    filename = corpus_dir / Path(f"{youtube_id}.wav")

    recording = Recording.from_file(filename, recording_id=youtube_id)
    segments = []
    for annotation in item:
        _, start_time, end_time, label = annotation.strip().split("\t")
        try:
            segment = SupervisionSegment(
                id=f"{youtube_id}-{start_time}-{end_time}-{label}",
                recording_id=youtube_id,
                start=float(start_time),
                duration=float(end_time) - float(start_time),
                channel=0,
                text=lookup[label],
            )
            segments.append(segment)
        except KeyError:
            return None, None
    return recording, segments
