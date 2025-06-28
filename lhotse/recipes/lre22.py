import torchaudio.functional.vad as vad
from pathlib import Path
from typing import Dict, Optional, Union
from tqdm import tqdm
from lhotse.audio import Recording, RecordingSet
from lhotse.parallel import parallel_map
from lhotse.supervision import SupervisionSegment, SupervisionSet


def prepare_lre22(
    corpus_dir: Pathlike,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 4,
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    pass


