#!/usr/bin/env python3
# Streaming BAM -> Feather CpG extractor (multiprocessing, low memory)
# Drops aligned-pairs immediately (no get_aligned_pairs()) and writes Feather via Arrow IPC streaming.
#
# Reduced output schema matches MethyLYZR expectations:
# epic_id: string
# methylation: double (0..1)
# scores_per_read: int64
#
# Storage note:
# - To minimize output size, ONLY the first 3 MethyLYZR-required columns are written:
#     epic_id (string), methylation (float 0..1), scores_per_read (int)
# - All other fields (read_id, run_id, start_time, QS, read_length, map_qs, binary_methylation) are intentionally omitted.
# binary_methylation: int64
# read_id: string
# start_time: int64   (seconds since run start DT in BAM header)
# run_id: string
# QS: double
# read_length: int64
# map_qs: int64
#
# Key behavior changes vs the original MethyLYZR BAM->Feather script:
# - Fixes primary-alignment filtering: excludes secondary OR supplementary alignments using
#     if aln.is_secondary or aln.is_supplementary: continue
#   (avoids incorrect bitwise-operator logic).
# - Treats BOTH 5mC and 5hmC calls as "methylation" by merging modbase tags per query position
#   and taking the max probability across ('C', strand, 'm') and ('C', strand, 'h').
#
# Notes:
# - Requires: pysam, numpy, pyarrow
# - This implementation computes start_time as seconds from BAM RG[0]['DT'] (no extra tdif rounding).
# - Filters:
#   - primary alignments only (not secondary/supplementary/unmapped)
#   - MAPQ >= mapq_min
#   - keep modbase calls with p <= meth_upper OR p >= meth_lower
#   - CpGs per read must be <= max_cpgs_per_read (default 10) (reads exceeding this are skipped)
#
# Example:
#   python bam2feather_light.py \
#     -i /path/to/bams \
#     --sites EPIC_CpGannotation.bed \
#     -s SAMPLE -o out_feather_file \
#     --io_threads 4 --chunk_size 50000

import argparse
import os
import pathlib
import sys
import time
from collections import defaultdict
from multiprocessing import Process, Queue, Value

import numpy as np
import pysam
import pyarrow as pa
import pyarrow.ipc as ipc

# -------------------------
# BED loading (fast maps)
# -------------------------

def load_cpg_maps(bed_path):
    """
    Returns:
      start_map[chrom][start] = epic_id
      end_map[chrom][end_minus_1] = epic_id   (matches original code's end = end - 1)
    """
    start_map = defaultdict(dict)
    end_map = defaultdict(dict)

    with open(bed_path, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom, start, end, epic_id = parts[:4]
            start_i = int(start)
            end_i = int(end) - 1
            start_map[chrom][start_i] = epic_id
            end_map[chrom][end_i] = epic_id

    return start_map, end_map

# -------------------------
# CIGAR-based mapping
# -------------------------

_CONSUMES_QUERY = {0, 1, 4, 7, 8}
_CONSUMES_REF   = {0, 2, 3, 7, 8}

def map_query_to_ref_positions(aln, query_positions_sorted):
    """
    Map selected query positions (0-based) to reference positions (0-based) without get_aligned_pairs().

    Returns dict {query_pos: ref_pos} only for positions mapping to a ref base.
    """
    if not query_positions_sorted:
        return {}

    cig = aln.cigartuples
    if cig is None:
        return {}

    q = 0
    r = aln.reference_start  # 0-based
    out = {}

    i = 0
    for op, length in cig:
        # insertion or softclip: consumes query only, no mapping to ref
        if op in (1, 4):
            q += length
            continue
        # deletion or ref skip: consumes ref only
        if op in (2, 3):
            r += length
            continue
        # match/equal/mismatch: consumes both; 1:1 mapping within block
        if op in (0, 7, 8):
            block_q_start = q
            block_q_end = q + length  # exclusive

            # advance i until inside block
            while i < len(query_positions_sorted) and query_positions_sorted[i] < block_q_start:
                i += 1

            while i < len(query_positions_sorted) and query_positions_sorted[i] < block_q_end:
                qp = query_positions_sorted[i]
                out[qp] = r + (qp - block_q_start)
                i += 1

            q += length
            r += length

            if i >= len(query_positions_sorted):
                break
            continue

        # hardclip/pad/other ops
        if op in _CONSUMES_QUERY:
            q += length
        if op in _CONSUMES_REF:
            r += length

    return out

# -------------------------
# File discovery
# -------------------------

def read_bam_files(path, recursive, file_queue, bams_amount, bam_filter):
    d = pathlib.Path(path)
    if recursive:
        files = list(d.rglob("*.bam"))
    else:
        files = list(d.glob("*.bam"))

    # optional filter
    if bam_filter is not None:
        files = [p for p in files if bam_filter in str(p)]

    files = sorted([str(p) for p in files])
    with bams_amount.get_lock():
        bams_amount.value = len(files)

    for fp in files:
        file_queue.put(fp)

# -------------------------
# Worker:  CpGs from BAM and send batches to writer
# -------------------------

def io_worker(
    file_queue,
    out_queue,
    sites_bed,
    mapq_min,
    meth_lower,
    meth_upper,
    max_cpgs_per_read,
    batch_rows,
    bams_analysed,
    rows_emitted,
):
    import datetime

    pid = os.getpid()
    print(f"[io_worker PID {pid}] started", flush=True)

    # load CpG maps in each worker (avoids huge pickles; OK on HPC)
    start_map, end_map = load_cpg_maps(sites_bed)

    def safe_tag(aln, tag, default=None):
        try:
            return aln.get_tag(tag)
        except Exception:
            return default

    while True:
        bamfile = file_queue.get()
        if bamfile is None:
            break

        try:
            bam = pysam.AlignmentFile(bamfile, "rb")
            header = bam.header.as_dict()
            rg0 = (header.get("RG") or [{}])[0]
            run_start = rg0.get("DT")
            run_id_hdr = rg0.get("ID")

            # Parse run_start timestamp (ISO 8601)
            run_start_dt = None
            if run_start is not None:
                # datetime.fromisoformat handles "+00:00"
                run_start_dt = datetime.datetime.fromisoformat(run_start.replace("Z", "+00:00"))

            batch = []
            local_rows = 0

            VALID_CHROMS = {
                "chr" + str(i) for i in range(1, 23)
            } | {"chrX", "chrY", "chrM"}

            for aln in bam:
                # primary + mapped + mapq filter
                if aln.is_unmapped:
                    continue
                if aln.is_secondary or aln.is_supplementary:
                    continue
                if aln.mapping_quality < mapq_min:
                    continue
                chrom = aln.reference_name
                if chrom is None:
                    continue
                if chrom not in VALID_CHROMS:
                    continue

                mb = aln.modified_bases
                if not mb:
                    continue
                
                # ---- merge 5mC + 5hmC by max per query position ----
                strand = 1 if aln.is_reverse else 0
                calls_m = dict(mb.get(("C", strand, "m"), []))  # {qpos: score 0..255}
                calls_h = dict(mb.get(("C", strand, "h"), []))  # {qpos: score 0..255}

                if not calls_m and not calls_h:
                    continue

                all_qpos = set(calls_m) | set(calls_h)

                # select confident calls early; build qpos list + merged probability dict
                qpos = []
                qprob = {}
                for qp in all_qpos:
                    pm = calls_m.get(qp, 0) / 255.0
                    ph = calls_h.get(qp, 0) / 255.0
                    p = pm if pm >= ph else ph  # TAKE MAX
                    if p <= meth_upper or p >= meth_lower:
                        qp_i = int(qp)
                        qpos.append(qp_i)
                        qprob[qp_i] = p

                if not qpos:
                    continue

                qpos.sort()
                q2r = map_query_to_ref_positions(aln, qpos)
                if not q2r:
                    continue

                cpg_map = end_map[chrom] if aln.is_reverse else start_map[chrom]
                if not cpg_map:
                    continue

                hits = []
                for qp in qpos:
                    rp = q2r.get(qp)
                    if rp is None:
                        continue
                    epic = cpg_map.get(rp)
                    if epic is None:
                        continue
                    hits.append((epic, qprob[qp]))

                if not hits:
                    continue

                # cap CpGs per read (consistent with predictor filtering)
                if len(hits) > max_cpgs_per_read:
                    #hits = hits[:max_cpgs_per_read]
                    continue

                scores_per_read = int(len(hits))

                # metadata (tags are per read)
                # st = safe_tag(aln, "st")
                # rg = safe_tag(aln, "RG", run_id_hdr)
                # qs = safe_tag(aln, "qs")

                # convert start_time to seconds since run start DT if possible, else -1
                # start_time_sec = -1
                # if st is not None and run_start_dt is not None:
                #     try:
                #         st_dt = datetime.datetime.fromisoformat(str(st).replace("Z", "+00:00"))
                #         start_time_sec = int((st_dt - run_start_dt).total_seconds())
                #     except Exception:
                #         start_time_sec = -1

                read_len = aln.infer_read_length()
                mapq = int(aln.mapping_quality)
                read_id = aln.query_name

                for epic_id, p in hits:
                    batch.append([
                        str(epic_id),
                        float(p),
                        scores_per_read,
                        # int(p >= 0.8),
                        # str(read_id),
                        # int(start_time_sec),
                        # str(rg) if rg is not None else (str(run_id_hdr) if run_id_hdr is not None else ""),
                        # float(qs) if qs is not None else np.nan,
                        # int(read_len) if read_len is not None else -1,
                        # mapq,
                    ])
                    local_rows += 1

                    if len(batch) >= batch_rows:
                        out_queue.put(batch)
                        batch = []

                        with rows_emitted.get_lock():
                            rows_emitted.value += batch_rows

            # flush remaining
            if batch:
                out_queue.put(batch)
                with rows_emitted.get_lock():
                    rows_emitted.value += len(batch)

            bam.close()

            with bams_analysed.get_lock():
                bams_analysed.value += 1

            print(f"[io_worker PID {pid}] done {bamfile} rows={local_rows}", flush=True)

        except Exception as e:
            print(f"[io_worker PID {pid}] ERROR on {bamfile}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # signal this worker finished
    out_queue.put(None)
    print(f"[io_worker PID {pid}] exiting", flush=True)

# -------------------------
# Writer:  batches to Feather (Arrow IPC file)
# -------------------------

def writer_process(out_queue, output_path, n_workers, done_flag):
    pid = os.getpid()
    print(f"[writer PID {pid}] started; output={output_path}", flush=True)

    schema = pa.schema([
        ("epic_id", pa.string()),
        ("methylation", pa.float64()),
        ("scores_per_read", pa.int64()),
        # ("binary_methylation", pa.int64()),
        # ("read_id", pa.string()),
        # ("start_time", pa.int64()),
        # ("run_id", pa.string()),
        # ("QS", pa.float64()),
        # ("read_length", pa.int64()),
        # ("map_qs", pa.int64()),
    ])

    # Ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


    finished_workers = 0
    rows_written = 0

    with pa.OSFile(output_path, "wb") as sink:
        with ipc.new_file(sink, schema) as writer:
            while True:
                item = out_queue.get()
                if item is None:
                    finished_workers += 1
                    print(f"[writer PID {pid}] worker finished ({finished_workers}/{n_workers})", flush=True)
                    if finished_workers >= n_workers:
                        break
                    continue

                # item is a list of rows
                cols = list(zip(*item))
                batch = pa.record_batch(
                    [
                        pa.array(cols[0], type=pa.string()),
                        pa.array(cols[1], type=pa.float64()),
                        pa.array(cols[2], type=pa.int64()),
                        # pa.array(cols[3], type=pa.int64()),
                        # pa.array(cols[4], type=pa.string()),
                        # pa.array(cols[5], type=pa.int64()),
                        # pa.array(cols[6], type=pa.string()),
                        # pa.array(cols[7], type=pa.float64()),
                        # pa.array(cols[8], type=pa.int64()),
                        # pa.array(cols[9], type=pa.int64()),
                    ],
                    schema=schema,
                )
                writer.write_batch(batch)
                rows_written += len(item)

                if rows_written % 200000 == 0:
                    print(f"[writer PID {pid}] rows_written={rows_written}", flush=True)

    with done_flag.get_lock():
        done_flag.value = 1

    print(f"[writer PID {pid}] finished rows_written={rows_written}", flush=True)

# -------------------------
# Progress reporter
# -------------------------

def progress(bams_analysed, bams_amount, rows_emitted, writer_done):
    while True:
        time.sleep(1)
        msg = f"\rBams: {bams_analysed.value}/{bams_amount.value}  RowsQueued: {rows_emitted.value}"
        print(msg, end="", flush=True)
        if writer_done.value == 1:
            print("", flush=True)
            break

# -------------------------
# Main
# -------------------------

def main(inputs, recursive, io_threads, sites, sample, output, bam_filter,
         mapq_min, meth_lower, meth_upper, max_cpgs_per_read, batch_rows):
    if not output.endswith(".feather"):
        raise ValueError("Output path must end with .feather")

    file_queue = Queue()
    out_queue = Queue(maxsize=io_threads * 4)

    bams_amount = Value("i", 0, lock=True)
    bams_analysed = Value("i", 0, lock=True)
    rows_emitted = Value("i", 0, lock=True)
    writer_done = Value("i", 0, lock=True)

    print("Start getting files.", flush=True)
    bam_process = Process(
        target=read_bam_files,
        args=(inputs, recursive, file_queue, bams_amount, bam_filter),
    )
    bam_process.start()

    # start writer first
    out_path = output
    wproc = Process(target=writer_process, args=(out_queue, out_path, io_threads, writer_done))
    wproc.start()

    print("Start reading BAMs (streaming).", flush=True)

    workers = []
    for _ in range(io_threads):
        p = Process(
            target=io_worker,
            args=(
                file_queue,
                out_queue,
                sites,
                mapq_min,
                meth_lower,
                meth_upper,
                max_cpgs_per_read,
                batch_rows,
                bams_analysed,
                rows_emitted,
            ),
        )
        p.start()
        workers.append(p)

    # progress reporter
    prog = Process(target=progress, args=(bams_analysed, bams_amount, rows_emitted, writer_done))
    prog.start()

    # wait for file discovery to finish, then send sentinels to workers
    bam_process.join()
    for _ in range(io_threads):
        file_queue.put(None)

    for p in workers:
        p.join()

    wproc.join()
    prog.join()

    print(f"Done. Wrote Feather: {out_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", type=str, default=".", required=True, help="Directory with BAM files")
    parser.add_argument("-r", "--recursive", default=False, action="store_true", help="Recursively search subdirectories")
    parser.add_argument("--io_threads", type=int, default=2, help="Number of BAM worker processes")
    parser.add_argument("--sites", type=str, required=True, help="CpG annotation BED (EPIC) with 4 columns")
    parser.add_argument("-s", "--sample", type=str, required=True, help="Sample name (output file prefix)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output Feather file (.feather)")
    parser.add_argument("--filter", type=str, default=None, help="Only process BAM paths containing this substring")

    # filtering / model-compat knobs
    parser.add_argument("--mapq_min", type=int, default=10, help="Minimum MAPQ")
    parser.add_argument("--methLowerBound", type=float, default=0.8, help="Keep p>=this")
    parser.add_argument("--methUpperBound", type=float, default=0.2, help="Keep p<=this")
    parser.add_argument("--max_cpgs_per_read", type=int, default=10, help="Cap CpGs emitted per read")
    parser.add_argument("--chunk_size", type=int, default=50000, help="Rows per batch sent to writer")

    args = parser.parse_args()

    main(
        inputs=args.inputs,
        recursive=args.recursive,
        io_threads=args.io_threads,
        sites=args.sites,
        sample=args.sample,
        output=args.output,
        bam_filter=args.filter,
        mapq_min=args.mapq_min,
        meth_lower=args.methLowerBound,
        meth_upper=args.methUpperBound,
        max_cpgs_per_read=args.max_cpgs_per_read,
        batch_rows=args.chunk_size,
    )
