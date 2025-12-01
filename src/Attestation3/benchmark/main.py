from collections.abc import Generator
from operator import itemgetter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import polars as pl
import sherpa_onnx
import soundfile as sf
import torch
from benchmark_utils import Segment, run_benchmark_for
from resemblyzer import VoiceEncoder
from silero_vad import get_speech_timestamps, load_silero_vad
from sklearn.cluster import AgglomerativeClustering


def resemblyzer_silero_vad(filepath: str, n_clusters: int | None = 2) -> Generator[Segment, Any, Any]:
    encoder = VoiceEncoder("cpu")
    model = load_silero_vad(onnx=True)
    wav_numpy, _ = librosa.load(filepath, sr=16000, mono=True)
    wav = torch.from_numpy(wav_numpy).float()
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        return_seconds=True,
    )

    embeddings = []
    valid_segments = []

    for start, end in map(itemgetter("start", "end"), speech_timestamps):
        start_sample = int(start * 16000)
        end_sample = int(end * 16000)

        segment = wav_numpy[start_sample:end_sample]

        if len(segment) < 6400:
            segment = np.pad(segment, (0, 6400 - len(segment)))

        try:
            emb = encoder.embed_utterance(segment)
            embeddings.append(emb)
            valid_segments.append({"start": start, "end": end})
        except Exception:
            continue

    if embeddings:
        embeddings_array = np.array(embeddings)  # shape: (N, 256)

        if len(embeddings_array) > 1:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                distance_threshold=0.90 if n_clusters is None else None,
            )
            labels = clustering.fit_predict(embeddings_array)

            for i, seg in enumerate(valid_segments):
                if seg["end"] - seg["start"] > 0.3:
                    yield {
                        "start": seg["start"],
                        "end": seg["end"],
                        "label": f"SPEAKER_{labels[i]:02d}",
                    }


def resample_audio(audio, sample_rate, target_sample_rate):
    if sample_rate != target_sample_rate:
        print(f"Resampling audio from {sample_rate}Hz to {target_sample_rate}Hz...")
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
        print(f"Resampling completed. New audio shape: {audio.shape}")
        return audio, target_sample_rate
    return audio, sample_rate


def init_speaker_diarization(embedding_extractor_model: str, num_speakers: int = -1, cluster_threshold: float = 0.5):
    segmentation_model = "./sherpa-onnx-pyannote-segmentation-3-0.onnx"

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation_model,
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=embedding_extractor_model,
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold=cluster_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError(
            "Please check your config and make sure all required files exist",
        )

    return sherpa_onnx.OfflineSpeakerDiarization(config)


def progress_callback(num_processed_chunk: int, num_total_chunks: int) -> int:
    progress = num_processed_chunk / num_total_chunks * 100
    print(f"Progress: {progress:.3f}%")
    return 0


def sherpa_onnx_based(
    wave_filename: str,
    embedding_extractor_model: str,
    num_speakers: int = 2,
) -> Generator[Segment, Any, Any]:
    if not Path(wave_filename).is_file():
        msg = f"{wave_filename} does not exist"
        raise RuntimeError(msg)

    audio, sample_rate = sf.read(wave_filename, dtype="float32", always_2d=True)
    audio = audio[:, 0]

    sd = init_speaker_diarization(num_speakers=num_speakers, embedding_extractor_model=embedding_extractor_model)

    target_sample_rate = sd.sample_rate
    audio, sample_rate = resample_audio(audio, sample_rate, target_sample_rate)

    if sample_rate != sd.sample_rate:
        msg = f"Expected samples rate: {sd.sample_rate}, given: {sample_rate}"
        raise RuntimeError(
            msg,
        )

    show_progress = False

    if show_progress:
        result = sd.process(audio, callback=progress_callback).sort_by_start_time()
    else:
        result = sd.process(audio).sort_by_start_time()

    for r in result:
        yield {
            "start": r.start,
            "end": r.end,
            "label": f"speaker_{r.speaker:02}",
        }


vals = [
    run_benchmark_for(resemblyzer_silero_vad, "resemblyzer + silero_vad"),
    run_benchmark_for(
        lambda path: sherpa_onnx_based(path, "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"),
        "pyannote-segmentation-3-0 + 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced",
    ),
    run_benchmark_for(
        lambda path: sherpa_onnx_based(path, "voxblink2_samresnet34_ft.onnx"),
        "pyannote-segmentation-3-0 + voxblink2_samresnet34_ft",
    ),
    run_benchmark_for(
        lambda path: sherpa_onnx_based(path, "voxblink2_samresnet100_ft.onnx"),
        "pyannote-segmentation-3-0 + voxblink2_samresnet100_ft",
    ),
    run_benchmark_for(
        lambda path: sherpa_onnx_based(path, "voxceleb_gemini_dfresnet114_LM.onnx"),
        "pyannote-segmentation-3-0 + voxceleb_gemini_dfresnet114_LM",
    ),
    run_benchmark_for(
        lambda path: sherpa_onnx_based(path, "wespeaker_en_voxceleb_resnet293_LM.onnx"),
        "pyannote-segmentation-3-0 + wespeaker_en_voxceleb_resnet293_LM",
    ),
    run_benchmark_for(
        lambda path: sherpa_onnx_based(path, "nemo_en_titanet_large.onnx"),
        "pyannote-segmentation-3-0 + nemo_en_titanet_large",
    ),
]

df = pl.DataFrame([v.__dict__ for v in vals]).write_excel("benchmark_results.xlsx")
