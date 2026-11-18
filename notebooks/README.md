# Audiobench Cloud & Memory-Safe Transcriber

This directory contains the universal, reproducible Jupyter / Google Colab transcription notebook [`transcriptions.ipynb`](transcriptions.ipynb).

It enables high-speed, cost-free transcription of massive audiobooks and lectures (up to 24+ hours) using free GPU hardware (Google Colab T4/A100, Kaggle, RunPod, or local GPUs) with zero system RAM crashes.

---

## Key Features

1. **Memory-Safe 20-Minute Chunking**:
   * Slices audio on disk with `ffmpeg` into 20-minute chunks (`chunk_length_sec=1200`) with a 30-second overlap (`overlap_sec=30`).
   * Eliminates the fatal CPU RAM OOM crash when PyAV decodes multi-gigabyte audiobooks into memory at once.
2. **Word-Level LCS Stitcher (`stitch_chunk_segments`)**:
   * Automatically aligns the word-level stream across overlapping boundaries using Longest Common Subsequence (LCS).
   * Excises repeated paragraphs and guarantees **strictly monotonic timestamps** ($S_{i+1}.\text{start} \ge S_{i}.\text{end}$).
   * Automatically discards buffer-edge truncation artifacts (`The Crap / The Crap` loops) caused by mid-syllable slicing.
3. **Audiobench Schema Version 1 Compliant**:
   * Computes the SHA-256 audio file hash matching Audiobench's internal deduplication specification.
   * Produces `.json` files ready for instant, 0-second ingestion via `audiobench import --transcript`.
4. **Environment Hardening**:
   * Pre-loads CUDA 12 shared libraries (`libcublas.so`, `libcudnn.so`) dynamically into memory via `ctypes`.
   * Applies an in-memory recursion-proof PyAV patch to prevent `metadata_errors` crashes on corrupted ID3 tags.
   * Emits live human-readable timestamp previews (`[HH:MM:SS -> HH:MM:SS]`).

---

## Why 20 Minutes (1,200s) with 30s Overlap?

| Parameter | Recommended Value | Rationale |
| :--- | :--- | :--- |
| **`chunk_length_sec`** | `1200` (20 mins) | 20 minutes of 16kHz mono audio is only ~38 MB of PCM in RAM. Slicing every 20 minutes keeps RAM flat while allowing Whisper to run at peak throughput. |
| **`overlap_sec`** | `30` seconds | 30 seconds contains ~75 spoken words. This provides an unambiguous, robust sequence for the LCS matcher to align across chunks without wasting unnecessary GPU inference cycles. |

### What If I Use Different Durations?
* **10 Minutes (`chunk_length_sec=600`)**: Safe for very low-memory environments (e.g. 4 GB VPS), but slightly increases overall transcription time due to more chunk overhead.
* **30–45 Minutes (`chunk_length_sec=2700`)**: Works on machines with $\ge 16$ GB RAM.
* **Overlap Invariance**: The LCS stitcher is mathematically invariant to chunk length and overlap range—it extracts and aligns whatever words fall into $[T_{\text{offset}} - 2s, T_{\text{end}} + 2s]$ dynamically.

---

## How to Run in Google Colab

1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com).
2. Click **File** $\rightarrow$ **Upload notebook** and upload [`transcriptions.ipynb`](transcriptions.ipynb).
3. Set your runtime to GPU: **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ **T4 GPU** (or A100).
4. Run the cells sequentially:
   * **Step 1**: Mounts Google Drive to persist all output.
   * **Step 2**: Installs `faster-whisper` and pre-loads CUDA libraries.
   * **Step 3**: Loads `large-v3-turbo` (or `large-v3`).
   * **Step 4**: Defines the memory-safe overlapping transcriber and word-level LCS stitcher.
   * **Step 5**: Automatically detects media files in `/content` and transcribes them sequentially.
   * **Step 6**: Packages all resulting JSONs and transcripts into a zip file for instant download.

---

## Importing Back into Audiobench

Once downloaded to your local machine:
```bash
audiobench import --transcript
```
Audiobench will match the companion transcripts to your audio files, verify the SHA-256 hashes, and import them with 100% word-level precision into your Knowledge Graph.
