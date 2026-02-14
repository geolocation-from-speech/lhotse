from lhotse import CutSet, Seconds
from lhotse.lazy import Dillable
from lhotse.utils import fastcopy, uuid4
from lhotse.cut.mixed import MixedCut, MixTrack
from lhotse import MonoCut, MultiCut
from itertools import chain

import numpy as np
import math
from itertools import zip_longest, groupby
import warnings
import random
from typing import Optional, Sequence, List
import time


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
        min_unique: int = 1,
        max_num_overlaps: int = 5,
        max_snr: List[float] = None,
        sampling_rate: int = 16000,
        normalize_loudness: bool = False,
        reverb: bool = False,
        max_splices_schedule_increment: float = 4e-05,
        min_splices_schedule_increment: float = 4e-05,
        max_duration_increment: float = 3e-04,
        final_max_splices: Optional[int] = 6,
        final_min_splices: Optional[int] = 2,
        drift: float = 0.1,
        overlap: float = 0.5,
        min_overlap: float = 0.0,
        self_overlap: bool = False,
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
        self.min_unique = min_unique
        self.min_splices = min_splices
        self.max_num_overlaps = max_num_overlaps
        assert min_splices <= max_splices
        self.normalize_loudness = normalize_loudness
        self.reverb = reverb
        self.drift = drift
        self.self_overlap = self_overlap
        self.overlap = overlap
        self.min_overlap = min_overlap
        assert overlap >= min_overlap
        if max_snr is not None:
            assert len(max_snr) == len(cutsets)
            self.max_snr = max_snr
        else:
            self.max_snr = [0.0 for i in cutsets]

        self.sr = sampling_rate
        self.seed = seed
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

    def next_rand(self, cs_idx, max_steps=10):
        k = random.randint(1, max_steps)
        numels = 0
        for _ in range(k):
            try:
                cut_to_append = next(self.cutset_iterables[cs_idx])
            except StopIteration:
                self.cutset_iterables[cs_idx] = iter(self.cutsets[cs_idx])
                cut_to_append = next(self.cutset_iterables[cs_idx])
        return cut_to_append


    def concat_resample_and_norm(self, cuts, offsets, spk=None):
        if self.normalize_loudness:
            tracks = [
                MixTrack(
                    cut=cut.normalize_loudness(-23).resample(self.sr),
                    offset=offset
                )
                for cut, offset in zip(cuts, offsets)
            ]
        else:
            tracks = [
                MixTrack(cut=cut.resample(self.sr), offset=offset)
                for cut, offset in zip(cuts, offsets)
            ]
        concatenated = MixedCut(
            id=str(uuid4()),
            tracks=tracks
        )
        return concatenated

    def mix_cuts_with_snrs(
        self,
        cuts,
        snrs,
        offsets=None,
        reference_idx=0,
    ):
        """
        Mix cuts with specified global SNRs (relative to one reference).
    
        Args:
            cuts: list[Cut]
            snrs: list[float or None]
                  snrs[i] is the SNR (in dB) of cuts[i] relative to the reference.
                  Use None for the reference cut.
            offsets: list[float] or None
                     Offsets in seconds. Defaults to all zeros.
            reference_idx: int
                           Index of the reference cut.
        Returns:
            MixedCut
        """
        assert len(cuts) == len(snrs)
        n = len(cuts)
    
        if offsets is None:
            offsets = [0.0] * n
        assert len(offsets) == n
    
        def expand_tracks(cut, offset=0.0, snr=None):
            """
            Replicates mix() flattening semantics:
            - If cut is a structural MixedCut (no transforms), flatten its tracks
            - Otherwise, treat cut as atomic
            """
            if isinstance(cut, MixedCut) and len(cut.transforms or []) == 0:
                tracks = []
                for track in cut.tracks:
                    tracks.append(
                        MixTrack(
                            cut=track.cut,
                            offset=round(track.offset + offset, 8),
                            snr=(
                                track.snr
                                if snr is None
                                else snr
                                if track.snr is None
                                else track.snr + snr
                            ),
                        )
                    )
                return tracks
            else:
                return [MixTrack(cut=cut, offset=offset, snr=snr)]
    
        tracks = []
    
        # Reference cut (no SNR)
        tracks.extend(
            expand_tracks(
                cuts[reference_idx],
                offset=offsets[reference_idx],
                snr=None,
            )
        )
    
        # Other cuts
        for i, cut in enumerate(cuts):
            if i == reference_idx:
                continue
            tracks.extend(
                expand_tracks(
                    cut,
                    offset=offsets[i],
                    snr=snrs[i],
                )
            )
    
        return MixedCut(
            id=str(uuid4()),
            tracks=tracks,
        ) 
    
    def __iter__(self):
        self.cutset_iterables = {i: iter(cs) for i, cs in enumerate(self.cutsets)}
        return self

    def __next__(self):
        start_time = time.time()
        # Sample the number of splices
        # The number is random up to max_slices
        rng = random.Random()
        
        # Number of speakers
        num_unique_sets = rng.randint(self.min_unique, self.max_unique)
        # Number of turns total
        num_splices = rng.randint(self.min_splices, self.max_splices)
        
        # Select cutsets (speakers) from which to sample (possibly repeated)
        # Pick without replacement
        spliceable_cutsets = sample(
            self.cutset_weights, k=num_unique_sets, rng=rng
        )
        
        # Uniform
        cutsets_to_splice = rng.choices(
            spliceable_cutsets,
            k=num_splices,
        )

        # To encourage seeing different speakers we will group by value
        cutsets_groups = {k: list(g) for k, g in groupby(sorted(cutsets_to_splice))}
        cutsets_to_splice = []
        num_splices_counter = 0
        while num_splices_counter < num_splices:
            for k in cutsets_groups:
                try:
                    val = cutsets_groups[k].pop()
                    cutsets_to_splice.append(val)
                except IndexError:
                    continue 
                
                num_splices_counter += 1
                if num_splices_counter == num_splices:
                    break

        # What snr should be used? Choose one snr per spk and keep it
        # consistent for the resulting spliced cut
        snrs = {
            cs_idx: self.max_snr[cs_idx] * rng.random()
            for cs_idx in spliceable_cutsets
        }

        # Initialize allowable start times for each speaker
        start_times = {cs_idx: 0 for cs_idx in spliceable_cutsets}
        global_end_time = 0

        # Initialize the splice cut for each speaker
        splice_cuts = {}
        offsets = {}
        
        # Start creating overlapped / spliced audio samples
        for i, cs_idx in enumerate(cutsets_to_splice): 
            # Where should the segment start? We assume that segments occur as
            # in a conversation and each subsequent segment is increasingly
            # likely to happen later in the conversation. self.drift controls
            # how rapidly (on average) a new turn (possibly overlapped) starts.
            #
            # The self.self_overlap parameter determines if a speaker is
            # allowed to overlap with himself. The drift controls how much the
            # constructed utterance group moves left-to-right or whether it is
            # increasingly overlapped. It should be set higher in these cases 
            # to ensure there isn't too much overlap, especially if
            # self.max_splices is relatively high.
            if self.self_overlap:
                local_start = i * (self.drift * rng.random())
            else:
                local_start = start_times[cs_idx] + self.drift * rng.random()
            
            start_ = min(
                global_end_time,
                local_start,
            )
            # start = start_ + (1 - self.overlap*rng.random()) * (global_end_time - start_)
            lambda_ = rng.random()
            start = global_end_time - (global_end_time - start_)*(lambda_*self.overlap + (1-lambda_)*self.min_overlap)
            
            # ---------------- Initialization ------------------------
            # Get a tentative new cut
            next_cut = self.next_rand(cs_idx).resample(self.sr)
            
            # Initialize the speaker's track, i.e., splice_cut
            if cs_idx not in splice_cuts:
                splice_cuts[cs_idx] = []
                offsets[cs_idx] = []
           
            # There must be at least one cut 
            if i == 0:
                splice_cuts[cs_idx].append(next_cut)
                offsets[cs_idx].append(start)
                start_times[cs_idx] = start + next_cut.duration 
                global_end_time = max(global_end_time, start_times[cs_idx])
                continue
             
            # Check duration condition
            new_duration = start + next_cut.duration
            max_duration = self.max_duration
            
            # Check the number of overlaps
            num_overlaps = 0
            all_curr_cuts_and_offsets = []
            for cs_idx_ in splice_cuts:
                for s, o in zip(splice_cuts[cs_idx_], offsets[cs_idx_]):
                    all_curr_cuts_and_offsets.append((s, o))
            all_curr_cuts_and_offsets.append((next_cut, start))
            
            # Find the maximum number of overlapping supervisions in the new cut
            max_num_overlaps = max_overlap_with_offset(all_curr_cuts_and_offsets)
            
            # --------------- Stopping Condition ----------------------------
            # Return the constructed cut if these conditions are met
            if (
                (self.max_duration and new_duration > self.max_duration) or
                (max_num_overlaps > self.max_num_overlaps)
            ):
                cut_order = sorted(splice_cuts)
                for cs_idx_ in cut_order:
                    splice_cuts[cs_idx_] = self.concat_resample_and_norm(
                        splice_cuts[cs_idx_], offsets=offsets[cs_idx_],
                    )
                final_cut = self.mix_cuts_with_snrs(
                    [splice_cuts[cs_idx_] for cs_idx_ in cut_order],
                    [snrs[cs_idx_] for cs_idx_ in cut_order],
                )
                
                # Normalize the final segment for loudness if requested
                if self.normalize_loudness:
                    final_cut = final_cut.normalize_loudness(-23)
                if self.reverb:
                    # Don't reverberate all of them
                    if rng.random() > 0.5:
                        final_cut = final_cut.reverb_rir()
                return final_cut
            # ---------------------------------------------------------------
            
            # Otherwise keep constructing the cut
            splice_cuts[cs_idx].append(next_cut)

            # Keep track of the offset and start time
            offsets[cs_idx].append(start)
            start_times[cs_idx] = start + next_cut.duration 
                
            # Keep track of the global end time and start time 
            global_end_time = max(global_end_time, start_times[cs_idx])
            
        # -------------------------- End of for loop ------------------------- #
        # Flush splice_cuts
        cut_order = sorted(splice_cuts) 
        for cs_idx in cut_order:
            splice_cuts[cs_idx] = self.concat_resample_and_norm(
                splice_cuts[cs_idx], offsets=offsets[cs_idx],
            )
        final_cut = self.mix_cuts_with_snrs(
            [splice_cuts[cs_idx] for cs_idx in cut_order],
            [snrs[cs_idx] for cs_idx in cut_order],
        )
        
        if self.normalize_loudness:
            final_cut.normalize_loudness(-23)
        if self.reverb:
            # Don't reverberate all of them
            if rng.random() > 0.5:
                final_cut = final_cut.reverb_rir()

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
        return final_cut


# Gumbel-max trick for sampling
def sample(a, k=1, rng=None):
    values = [
        math.log(a_i) - math.log(-math.log(random.random())) for a_i in a
    ]
    return sorted(range(len(a)), key=lambda x: values[x], reverse=True)[:k]


def random_mono_cut(cut, fix_channel=None):
    if isinstance(cut, MonoCut):
        return cut
    elif isinstance(cut, MultiCut):
        if fix_channel is None:
            channel = random.choice(cut.channel)
        else:
            channel = fix_channel

        return cut.to_mono()[channel]
    else:
        raise ValueError(f"Unexpected cut type: {type(cut)}")


def max_overlap_with_offset(cuts_with_offsets):
    """
    cuts_with_offsets: list of (cut, offset) pairs
        - cut: a Lhotse Cut (may be None)
        - offset: float, number of seconds to shift all supervisions
    
    Returns:
        int: maximum number of overlapping supervisions across all cuts.
    """
    events = []
    for cut, offset in cuts_with_offsets:
        if cut is None:
            continue
        for sup in cut.supervisions:
            start = sup.start + offset
            end = start + sup.duration
            events.append((start, +1))
            events.append((end, -1))

    if not events:
        return 0

    cur, max_ov = 0, 0
    for _, delta in sorted(events):
        cur += delta
        max_ov = max(max_ov, cur)

    return max_ov


def test(num_iters):
    from pathlib import Path
    from lhotse import load_manifest_lazy 
    root = "/ocean/projects/cis210027p/mwiesner/jsalt2025/icefall/egs/librispeech/MTASR"
    #cut_manifests = list((Path(root) / Path("data/manifests")).rglob("cuts*train*shuffled_*.jsonl.gz"))
    #cutsets = []
    #for cm in cut_manifests:
    #    cutsets.append(load_manifest_lazy(cm))
    #snrs = [0, 0, 0, 0]
    #for p, s in zip(cut_manifests, snrs):
    #    print(f'{p}: {s}')
    speakers = list(Path(f"{root}/data/manifests/librispeech_speakers").rglob("*.jsonl.gz"))
    num_spks = len(speakers) 
    weight = 1/num_spks
    sim_weight = 0.0
    cut_info = []
    for s in speakers:
        cut_info.append((s.stem.split("_")[-1].split(".")[0], (1-sim_weight)*weight, s))
    
    cutsets, weights, names = [], [], []
    for n, w, p in cut_info: 
        cutsets.append(load_manifest_lazy(p))
        weights.append(w)
        names.append(n)

    cs_iter = CutSpliceIterable(
        cutsets,
        max_splices=10,
        min_splices=2,
        max_duration=60,
        final_max_duration=60,
        max_unique=3,
        min_unique=3,
        max_num_overlaps=3,
        sampling_rate=16000,
        max_snr=[60 * (random.random() - 0.5) for i in range(num_spks)],
        cutset_weights=weights,  
        normalize_loudness=False,
        final_min_splices=2,
        final_max_splices=10,
        max_splices_schedule_increment=1e-02,
        min_splices_schedule_increment=1e-02,
        max_duration_increment=1e-01,
        reverb=True,
        drift=2,
        overlap=1.0,
        min_overlap=0.2,
        self_overlap=True,
    )
    for i in range(num_iters):
        c = next(cs_iter)
        try:
            audio = c.load_audio()
        except IndexError:
            import pdb; pdb.set_trace() 
        
        if i % 1 == 0:
            print(f'====={i}: {len(c.supervisions)}======')
            print(f"dur: {c.duration}")
            print(audio.shape)
            for s in c.supervisions:
                print(s)
                print('---')
