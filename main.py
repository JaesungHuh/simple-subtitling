from typing import List, Optional, Tuple, Dict, NamedTuple
import argparse
import os

import srt
import numpy as np
import tqdm
import whisperx
import torchaudio
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_wespeaker_model, write_subtitle

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Segment(NamedTuple):
    start: float
    end: float
    text: str
    speaker: str = "unknown"


def parse_exemplar_dir(exemplar_dir: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """Parse exemplar directory into a mapping of speaker names to their audio files."""
    """The structure of the exemplar_dir should be like below.
    exemplar_dir/
        spk_1/
            spk_1_1.wav
            spk_1_2.wav
            ...
        spk_2/
            spk_2_1.wav
            spk_2_2.wav
            ...
        ...
    """
    if not exemplar_dir:
        return None
    return {
        spk_dir: [os.path.join(exemplar_dir, spk_dir, f) for f in os.listdir(os.path.join(exemplar_dir, spk_dir))]
        for spk_dir in os.listdir(exemplar_dir)
    }


def transcribe(
    input_file: str,
    model_type: str = "medium",
    language: Optional[str] = None,
) -> List[Segment]:
    """Transcribe audio and return segments with timestamps and text."""
    model = whisperx.load_model(model_type, device, compute_type="int8", language=language)
    audio = whisperx.load_audio(input_file)
    result = model.transcribe(audio, batch_size=8)  # Change the batch size if you want to
    
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result_align = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    return [
        Segment(s["start"], s["end"], s["text"].strip())
        for s in result_align["segments"]
        if s["end"] - s["start"] > 0
    ]


def extract_embeddings(
    segments: List[Segment],
    input_file: str,
    exemplars: Optional[Dict[str, List[str]]] = None,
    model_type: str = "wespeaker",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
    """Extract speaker embeddings using either WeSpeaker or ECAPA-TDNN model."""
    signal, fs = torchaudio.load(input_file)
    signal = signal.to(device)
    
    # Initialize model
    if model_type == "ecapatdnn":
        from speechbrain.inference.speaker import EncoderClassifier
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        def get_embedding(audio):
            return np.squeeze(model.encode_batch(audio).detach().cpu().numpy())
    else:  # wespeaker
        import wespeaker
        model = wespeaker.load_model_local(get_wespeaker_model())
        model.set_device(device)
        def get_embedding(audio):
            return model.extract_embedding_from_pcm(audio, fs).detach().cpu().numpy()
    
    # Extract embeddings for segments
    embeddings = []
    for seg in tqdm.tqdm(segments, desc="Extracting embeddings"):
        audio_seg = signal[:, int(seg.start * fs):int(seg.end * fs)]
        if audio_seg.shape[1] < fs * 0.5:  # Pad short segments
            audio_seg = torch.nn.functional.pad(audio_seg, (0, int(fs * 0.5) - audio_seg.shape[1]))
        embeddings.append(get_embedding(audio_seg))
    embeddings = np.stack(embeddings, axis=0)
    
    # Process exemplars if provided
    if not exemplars:
        return embeddings, None, None
        
    embeddings_exemplar = []
    spk_list = []
    for spk_name, spk_files in exemplars.items():
        spk_embeddings = []
        for spk_file in spk_files:
            signal, _ = torchaudio.load(spk_file)
            spk_embeddings.append(get_embedding(signal))
        embeddings_exemplar.append(np.mean(spk_embeddings, axis=0))
        spk_list.append(spk_name)
    
    return embeddings, np.stack(embeddings_exemplar, axis=0), spk_list


def cluster_segments(
    segments: List[Segment],
    embeddings: np.ndarray,
    n_cluster: Optional[int] = None,
    distance_threshold: float = 0.8,
) -> List[Segment]:
    """Cluster segments by speaker embeddings."""
    clustering = AgglomerativeClustering(
        n_clusters=n_cluster,
        metric="cosine",
        linkage="average",
        distance_threshold=None if n_cluster else distance_threshold
    ).fit_predict(embeddings)
    
    return [
        Segment(seg.start, seg.end, seg.text, f"spk_{clustering[i]}")
        for i, seg in enumerate(segments)
    ]


def assign_speakers(
    segments: List[Segment],
    embeddings: np.ndarray,
    embeddings_exemplar: np.ndarray,
    spk_list: List[str],
    threshold: float,
) -> List[Segment]:
    """Assign speakers to segments based on exemplar embeddings."""
    similarity = cosine_similarity(embeddings, embeddings_exemplar)
    labels = np.argmax(similarity, axis=1)
    max_sims = np.max(similarity, axis=1)
    print(labels, max_sims)
    return [
        Segment(seg.start, seg.end, seg.text, spk_list[labels[i]] if max_sims[i] > threshold else "unknown")
        for i, seg in enumerate(segments)
    ]


def main(args: argparse.Namespace) -> None:
    """Main entry point for diarization and subtitle generation."""
    # Transcribe audio
    segments = transcribe(args.input_file, args.whisper_model_type, args.language)
    
    # Get exemplars if provided
    exemplars = parse_exemplar_dir(args.exemplar_dir)
    
    # Extract embeddings
    embeddings, embeddings_exemplar, spk_list = extract_embeddings(
        segments,
        args.input_file,
        exemplars,
        args.embedding_model
    )
    
    # Process segments
    if exemplars:
        segments = assign_speakers(
            segments,
            embeddings,
            embeddings_exemplar,
            spk_list,
            args.exemplar_threshold
        )
    else:
        segments = cluster_segments(
            segments,
            embeddings,
            args.n_cluster,
            args.distance_threshold
        )
    
    # Write subtitle
    write_subtitle(segments, out_file=args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker diarization using WhisperX and speaker embeddings")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input audio file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output srt file")
    parser.add_argument("--whisper_model_type", type=str, default="medium", help="Type of whisper model")
    parser.add_argument("--language", type=str, default=None, help="Language code for transcription")
    parser.add_argument("--embedding_model", type=str, default="wespeaker", choices=["ecapatdnn", "wespeaker"],
                       help="Type of speaker embedding model")
    parser.add_argument("--n_cluster", type=int, default=None, help="Number of speakers")
    parser.add_argument("--distance_threshold", type=float, default=0.8,
                       help="Distance threshold for clustering when n_cluster is None")
    parser.add_argument("--exemplar_dir", type=str, default=None, help="Path to audio exemplars")
    parser.add_argument("--exemplar_threshold", type=float, default=0.2,
                       help="Threshold for assigning unknown speakers (cosine similarity)")
    
    main(parser.parse_args())