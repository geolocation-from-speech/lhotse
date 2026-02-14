from pathlib import Path
from typing import Dict, Optional, Union, Sequence
import json

from lhotse.utils import Pathlike
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.audio import Recording, RecordingSet
from lhotse import CutSet, fix_manifests

from tqdm import tqdm


def prepare_librispeechmix(
    data_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    sets: Optional[Union[str, Sequence[str]]] = None,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(data_dir)

    manifests = {}
    
    listdir = data_dir / "list"
    # Do all the test sets if a subset is not specified
    if sets is None:
        set_paths = list(data_dir.rglob("*.jsonl"))
    else:
        set_paths = []
        if isinstance(sets, str):
            sets = [sets]
        for s in sets:
            set_paths.append(listdir / f"{s}.jsonl")

    for l in tqdm(set_paths, "Processing datasets ..."):
        datasetname = l.stem
        setdir = data_dir / "data" / datasetname
        sups = []
        recos = []
        with open(l, 'r') as f:
            str_data = f.readlines()
        for d_ in tqdm(str_data, f"Processing {datasetname} lines ..."):
            d = json.loads(d_)
            recording_id = Path(d['id']).stem
            reco_path = (data_dir / "data" / Path(d['mixed_wav'])).resolve()
            recos.append(Recording.from_file(reco_path))
            for i, (offset, duration, speaker, gender, text) in enumerate(
                zip(
                    d['delays'], d['durations'], d['speakers'],
                    d['genders'], d['texts'],
                )
            ):
                sups.append(
                    SupervisionSegment(
                        id=f"{recording_id}-{i}",
                        recording_id=recording_id,
                        start=offset,
                        duration=duration,
                        speaker=speaker,
                        text=text,
                        gender=gender,
                        channel=0,
                    )
                )
        sups = SupervisionSet.from_segments(sups)
        recos = RecordingSet.from_recordings(recos)
        recos, sups = fix_manifests(recos, sups)            

        manifests[datasetname] = {
            'supervisions': sups,
            'recordings': recos,
        }
        
        if output_dir is not None:
            sups.to_file(output_dir / f"librispeechmix_{datasetname}_recordings.jsonl.gz")
            recos.to_file(output_dir / f"librispeechmix_{datasetname}_supervisions.jsonl.gz")
            
            cuts = CutSet.from_manifests(recordings=recos, supervisions=sups)
            cuts.to_file(output_dir / f"cuts_librispeechmix_{datasetname}.jsonl.gz")

    return manifests
