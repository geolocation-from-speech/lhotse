from lhotse import CutSet, Seconds
from lhotse.lazy import Dillable
from lhotse.utils import fastcopy
from lhotse.cut.set import mix

import math
import warnings
import random
from typing import Optional, Sequence, List


class CutSpliceIterable(Dillable):
    """
        Samples a tuple of cuts from a set cuts (one per element in the set)
        and splices the cuts together. max_splice_duration is the maximum
        length of the cut that can be used. max_splices is the largest number
        of segments that can be spliced together. The sampling from cutsets
        can be weighted by cutset_weights. If this isn't specified, the
        weights are assumed to be uniform.

        Options for serialize are 'speech', 'all', and 'none'.
    """
    def __init__(
        self,
        cutsets: Sequence[CutSet],
        cutset_weights: Optional[Sequence[CutSet]] = None,
        cutset_prefixes: Optional[List[str]] = None,
        max_duration: Seconds = None,
        final_max_duration: Seconds = None,
        max_splices: int = 2,
        min_splices: int = 2,
        max_unique: int = 3,
        max_num_overlaps: int = 5,
        max_overlap: List[float] = None,
        min_overlap: List[float] = None,
        serialize: str = 'speech',
        max_snr: List[float] = None,
        sampling_rate: int = 16000,
        normalize_loudness: bool = False,
        max_splices_schedule_increment: float = 4e-05,
        min_splices_schedule_increment: float = 4e-05,
        max_duration_increment: float = 3e-04,
        final_max_splices: Optional[int] = 6,
        final_min_splices: Optional[int] = 2,
        enforce_separate_spk: bool = True,
        seed: int = 0,
    ):

        self.cutsets = cutsets
        self.cutset_iterables = {i: iter(cs) for i, cs in enumerate(cutsets)}
        if not cutset_weights:
           cutset_weights = [1./len(cutsets) for i in cutsets]
        self.cutset_weights = cutset_weights
        self.cutset_prefixes = cutset_prefixes
        self.max_duration = max_duration
        self.final_max_duration = final_max_duration
        self.max_splices = max_splices
        self.max_unique = max_unique
        self.min_splices = min_splices
        self.max_num_overlaps = max_num_overlaps
        assert min_splices <= max_splices
        self.normalize_loudness = normalize_loudness
        self.serialize = serialize
        if max_overlap is not None:
           assert len(max_overlap) == len(cutsets)
           self.max_overlap = max_overlap
        else:
           self.max_overlap = [0.0 for i in cutsets]

        if min_overlap is not None:
            assert len(min_overlap) == len(cutsets)
            for i, o in enumerate(min_overlap):
                assert o <= max_overlap[i]
            self.min_overlap = min_overlap
        else:
            self.min_overlap = [0.0 for i in cutsets]

        if max_snr is not None:
            assert len(max_snr) == len(cutsets)
            self.max_snr = max_snr
        else:
            self.max_snr = [0.0 for i in cutsets]

        self.sr = sampling_rate
        self.seed = seed
        self.splice_cut = None
        self.init_max_splices = max_splices
        self.init_min_splices = min_splices
        if final_max_splices is None:
            final_max_splices = max_splices
        if final_min_splices is None:
            final_min_splices = min_splices
        if final_max_duration is None:
            final_max_duration = max_duration
        self.final_max_splices = final_max_splices
        self.final_min_splices = final_min_splices
        self.max_splices_schedule_increment = max_splices_schedule_increment
        self.min_splices_schedule_increment = min_splices_schedule_increment
        self.max_duration_increment = max_duration_increment
        self.max_curr_splice_increment = 0
        self.min_curr_splice_increment = 0

    def set_max_splices(self, val: int):
        self.max_splices = val

    def set_min_splices(self, val: int):
        self.min_splices = val

    def set_max_duration(self, val: float):
        self.max_duration = val

    def __iter__(self):
        self.cutset_iterables = {i: iter(cs) for i, cs in enumerate(self.cutsets)}
        return self

    def __next__(self):
        # Sample the number of splices
        # The number is random up to max_slices
        rng = random.Random()
        num_unique_sets = rng.randint(1, self.max_unique)
        num_splices = rng.randint(self.min_splices, self.max_splices)
        # Select which cutsets from which to sample (possibly repeated)
        # Pick without replacement
        spliceable_cutsets = sample(
            self.cutset_weights, k=num_unique_sets, rng=rng
        )

        #print(f"IDXs of chosen manifests: {spliceable_cutsets}")
        # Uniform
        cutsets_to_splice = rng.choices(
            spliceable_cutsets,
            k=num_splices,
        )
        #print(f"Order of manifests: {cutsets_to_splice}")

        # Initialize the cut to return. Either it will be None or it will be
        # a cut that was previously sampled that we didn't want to discard.
        # We keep track of the cuts that are spliced together (no overlap),
        # separately from the cuts that have overlap. They are then mixed
        # once the stopping condition (duration, number of segments) has been
        # met.
        splice_cut = fastcopy(self.splice_cut) if self.splice_cut else None
        overlap_cut = None
        # We need to find out the start time from which subsequent cuts will
        # be spliced or overlapped. There is an edgecase however when there are
        # no overlaps selected and you can't tell until the end of the splicing
        # whether or not this case actually will occur. So for this reason we
        # do not store the overlap segment like we do the splice_cut 
        start = 0
        prev_dur = 0 
        use_splice = False
        use_overlap = False
        if splice_cut is not None:
            start = splice_cut.duration
        
        # Start creating overlapped / spliced audio samples
        for i, cs_idx in enumerate(cutsets_to_splice):
            #print("++next")
            # How much should this next segment be overlapped?
            overlap = (self.max_overlap[cs_idx] - self.min_overlap[cs_idx]) * rng.random() + self.min_overlap[cs_idx]
            #print(f'idx: {i}, cs_idx: {cs_idx}, overlap: {overlap}')
            # What snr should be used? Only apply to overlapped segments
            snr = self.max_snr[cs_idx] * rng.random() if self.max_overlap[cs_idx] > 0 else 0.0
            
            
            if overlap == 0:
                use_splice = True
            else:
                use_overlap = True

            # ---------------- Initialization ------------------------
            # Initialize the splice cut if there is not overlap
            if overlap == 0 and not splice_cut:
                try:
                    splice_cut = next(
                        self.cutset_iterables[cs_idx]
                    ).resample(self.sr)
                # Restart the iterator if you exhaust it
                except StopIteration:
                    self.cutset_iterables[cs_idx] = iter(self.cutsets[cs_idx])
                    splice_cut = next(
                        self.cutset_iterables[cs_idx]
                    ).resample(self.sr)
                # Normalize the segment for loudness if requested
                if self.normalize_loudness:
                    splice_cut = splice_cut.normalize_loudness(-23)
                prev_dur = splice_cut.duration
                # Update the start of the new segment and pad if start > 0
                if start > 0:
                    splice_cut = splice_cut.pad(duration=start, direction='left')
                start = splice_cut.duration
                continue
            # Initialize the overlap cut if there is overlap
            elif overlap > 0 and not overlap_cut:
                try:
                    overlap_cut = next(
                        self.cutset_iterables[cs_idx]
                    ).resample(self.sr)
                except StopIteration:
                    self.cutset_iterables[cs_idx] = iter(self.cutsets[cs_idx])
                    overlap_cut = next(
                        self.cutset_iterables[cs_idx]
                    ).resample(self.sr)
                if self.normalize_loudness:
                    overlap_cut = overlap_cut.normalize_loudness(-23)
                prev_dur = overlap_cut.duration
                if start > 0:
                    overlap_cut = overlap_cut.pad(duration=start, direction='left')
                start = overlap_cut.duration
                continue

            # ---------------- Tentative new cut -----------------------------
            try:
                cut_to_append = next(self.cutset_iterables[cs_idx])
            except StopIteration:
                self.cutset_iterables[cs_idx] = iter(self.cutsets[cs_idx])
                cut_to_append = next(self.cutset_iterables[cs_idx])

            # Get the amount overlap
            if overlap > 0:
                #offset = max(
                #    0, start - overlap * cut_to_append.duration
                #)
                offset = max(0, start - overlap * prev_dur)
            else:
                offset = start
            
            #print(f'offset: {offset}')
            #print(f'duration: {cut_to_append.duration}')
            #print(f'delta duration: {cut_to_append.duration - overlap*cut_to_append.duration}')
            # Check duration condition
            new_duration = offset + cut_to_append.duration
            #print(f'new_duration: {new_duration}, {self.max_duration}')
            max_duration = self.max_duration
            num_overlaps = 0
            if splice_cut:
                num_overlaps = sum([len(w.supervisions) for w in splice_cut.cut_into_windows(0.5)])
            if overlap_cut:
                num_overlaps += sum([len(w.supervisions) for w in overlap_cut.cut_into_windows(0.5)])

            if self.max_duration and new_duration > self.max_duration or num_overlaps > self.max_num_overlaps:
                # Two cases: One for overlap, the other for no overlap
                if overlap > 0:
                    # Since we are tossing the overlap cut, we just set the
                    # self.splice_cut to None for the next segment
                    self.splice_cut = None
                else:
                    self.splice_cut = cut_to_append.resample(self.sr)
                    # Again normalize the loudness if that option is requested
                    if self.normalize_loudness:
                        self.splice_cut = self.splice_cut.normalize_loudness(-23)
                    if self.serialize == 'speech':
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", message="^.*merging overlapping supervisions.*$"
                            )
                            splice_cut = splice_cut.merge_supervisions(
                                merge_policy="keep_first",
                            )
                # Because the duration condition is met we need to now mix the
                # overlap and splice cuts so ...
                # Figure out what cuts we have and how/if to mix them
                if splice_cut is not None and overlap_cut is not None:
                    final_cut = mix(splice_cut, overlap_cut, offset=0, snr=1, allow_padding=True)
                elif splice_cut is not None:
                    final_cut = splice_cut
                elif overlap_cut is not None:
                    final_cut = overlap_cut
                
                # Serialize all supervisions if requested
                if self.serialize == 'all':
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="^.*merging overlapping supervisions.*$"
                        )
                        final_cut = final_cut.merge_supervisions(
                            merge_policy="keep_first",
                        )
                return final_cut


            # Maybe normalize loudness
            cut_to_append = cut_to_append.resample(self.sr)
            if self.normalize_loudness:
                cut_to_append = cut_to_append.normalize_loudness(-23)
            
            # If the overlap is none add the new cut to the current splice_cut
            # Else add it to the current overlap_cut
            if overlap == 0:
                splice_cut = mix(
                    splice_cut,
                    cut_to_append,
                    offset=offset,
                    snr=snr,
                    allow_padding=True,
                )
                start = splice_cut.duration
            else:
                overlap_cut = mix(overlap_cut,
                    cut_to_append,
                    offset=offset,
                    snr=snr,
                    allow_padding=True,
                )
                start = overlap_cut.duration
            prev_dur = cut_to_append.duration

        # -------------- End of for loop ---------- #
        # Some cleanup after the loop
        if self.serialize == 'speech':
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="^.*merging overlapping supervisions.*$"
                )
                splice_cut = splice_cut.merge_supervisions(
                    merge_policy="keep_first"
                )
        
        # Figure out what cuts we have an how/if to mix them
        if splice_cut is not None and overlap_cut is not None:
            final_cut = mix(splice_cut, overlap_cut, offset=0, snr=1, allow_padding=True)
        elif splice_cut is not None:
            final_cut = splice_cut
        elif overlap_cut is not None:
            final_cut = overlap_cut
       
        if self.serialize == 'all':
            # Really annoying warning due to a possible bug in lhotse overlap
            # function that causes small differences in supervision end and start
            # times to be caught as an overlap.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="^.*merging overlapping supervisions.*$"
                )
                final_cut = final_cut.merge_supervisions(
                    merge_policy="keep_first"
                )
        self.splice_cut = None
        
        # Make the splices longer / more overlapped as training continues
        if self.final_max_splices:
            self.max_curr_splice_increment += self.max_splices_schedule_increment
            self.set_max_splices(
                int(
                    min(
                        self.final_max_splices,
                        self.init_max_splices + self.max_curr_splice_increment
                    )
                )
            )
        if self.final_min_splices:
            self.min_curr_splice_increment += self.min_splices_schedule_increment
            self.set_min_splices(
                int(
                    min(
                        self.final_min_splices,
                        self.init_min_splices + self.min_curr_splice_increment
                    )
                )
            )
        if self.final_max_duration:
            self.set_max_duration(
                min(
                    self.final_max_duration,
                    self.max_duration + self.max_duration_increment,
                )
            )  
        #print(f"num texts: {len(final_cut.supervisions)}")
        #print(f"duration: {final_cut.duration}")
        return final_cut


def sample(a, k=1, rng=None):
    values = [
        math.log(a_i) - math.log(-math.log(random.random())) for a_i in a
    ]
    return sorted(range(len(a)), key=lambda x: values[x], reverse=True)[:k]



def test():
    from pathlib import Path
    from lhotse import load_manifest_lazy 
    root = "/ocean/projects/cis210027p/mwiesner/jsalt2025/icefall/egs/librispeech/MTASR"
    cut_manifests = list((Path(root) / Path("data/manifests")).rglob("cuts*train*shuffled_*.jsonl.gz"))
    cutsets = []
    for cm in cut_manifests:
        cutsets.append(load_manifest_lazy(cm))
    overlaps = [1, 1, 1, 1]
    min_overlaps = [0.8, 0.8, 0.8, 0.8]
    snrs = [0, 0, 0, 0]
    for p, o, s in zip(cut_manifests, overlaps, snrs):
        print(f'{p}: {o}, {s}')
    cs_iter = CutSpliceIterable(
        cutsets,
        max_splices=2,
        min_splices=2,
        max_duration=30,
        max_unique=4,
        sampling_rate=16000,
        serialize='none',
        max_overlap=overlaps,
        min_overlap=min_overlaps,
        max_snr=[0, 0, 0, 0],
        cutset_weights=[0.25, 0.25, 0.25, 0.25],  
        normalize_loudness=True,
    )
    for i in range(5):
        c = next(cs_iter)
        c.save_audio(f"test_wav{i}.wav")
        audio = c.load_audio()
        print(f'====={len(c.supervisions)}======')
        print(f"dur: {c.duration}")
        print(audio.shape)
        for s in c.supervisions:
            print(s)
            print('---')
