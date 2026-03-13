import pandas as pd
from collections import defaultdict
import re
from collections import defaultdict

GO_ID_REGEX = re.compile(r"GO:\d{7}")


# ---------------------SEGMENT TO PROTOTYPE MAPPING ---------------------#


def add_residue_to_prototype_mapping(segment_assignments, prototype_assignments):
    """
    Maps cluster IDs in segment_assignments to prototype IDs using prototype_assignments.

    Args:
        segment_assignments (pd.DataFrame): DataFrame with columns ['pdb_id', 'cluster_ids', ...]
        prototype_assignments (pd.DataFrame): DataFrame with columns ['pdb_id', 'segment_k', 'proto', ...]

    Returns:
        pd.DataFrame: Copy of segment_assignments with an additional 'protos' column containing list of prototype IDs.
    """
    pdb_id_segment_to_prototype = {}

    for _, row in prototype_assignments.iterrows():
        key = (row['pdb_id'], row['segment_k'])
        pdb_id_segment_to_prototype[key] = row['proto']

    out = segment_assignments.copy()

    def parse_and_map(row):
        cluster_ids = [
            int(x.strip())
            for x in row['cluster_ids'].split(',')
            if x.strip() != ''
        ]
        return [
            pdb_id_segment_to_prototype.get((row['pdb_id'], cid))
            for cid in cluster_ids
        ]

    out['protos'] = out.apply(parse_and_map, axis=1)
    return out


def get_residue_ids(df, pdb_id):
    """
    Returns list of residue IDs for a given pdb_id from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with columns ['pdb_id', 'residue_ids', ...]
        pdb_id (str): PDB identifier to look up.
    Returns:
        List[int]: List of residue IDs.
    """
    row = df[df["pdb_id"] == pdb_id].iloc[0]
    residue_ids = [
        int(x.strip())
        for x in row['residue_ids'].split(',')
        if x.strip() != ''
    ]
    return residue_ids


def get_proto_ids(df, pdb_id):
    """
    Returns list of prototype IDs for a given pdb_id from the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame with columns ['pdb_id', 'protos', ...]
        pdb_id (str): PDB identifier to look up.
    Returns:
        List[int]: List of prototype IDs.
    """
    row = df[df["pdb_id"] == pdb_id].iloc[0]
    return row['protos']


# ---------------------INTERPRO HELPERS ---------------------#


def parse_pdb_chain(pdb_id_str: str):
    """
    Examples:
      "3ONG-B_B" -> pdb="3ONG", chain="B"
      "1AD3-A_A" -> pdb="1AD3", chain="A"

    If chain can't be parsed, returns chain=None.
    """
    s = str(pdb_id_str)
    pdb = s[:4].upper()
    chain = None
    # try the common "3ONG-B_B" pattern
    if "-" in s:
        after = s.split("-", 1)[1]
        if after:
            chain = after[0].upper()
    return pdb, chain


def extract_interpro_intervals(interpro_json: dict, pdb4: str, chain: str | None):
    """
    Returns list of dict intervals:
      {
        "accession": "IPR....",
        "name": "...",
        "type": "domain" | "family" | ...,
        "start0": int,   # 0-based inclusive
        "end0": int,     # 0-based inclusive
      }

    Uses entry_protein_locations fragments (start/end; 1-based) -> converts to 0-based.
    Filters by matching chain if chain is provided.
    """
    pdb4 = pdb4.upper()
    if pdb4 not in interpro_json:
        return []

    intervals = []
    results = interpro_json[pdb4].get("results", []) or []
    for entry in results:
        md = entry.get("metadata", {})
        acc = md.get("accession", "NA")
        name = md.get("name", "NA")
        typ = md.get("type", "NA")

        structures = entry.get("structures", []) or []
        for st in structures:
            # filter chain
            st_chain = st.get("chain", None)
            if chain is not None and st_chain is not None and st_chain.upper() != chain.upper():
                continue
            locations = st.get("entry_protein_locations", []) or []
            for loc in locations:
                fragments = loc.get("fragments", []) or []
                for frag in fragments:
                    # InterPro uses 1-based inclusive coords in these fragments
                    s = frag.get("start", None)
                    e = frag.get("end", None)
                    if s is None or e is None:
                        continue
                    start0 = int(s) - 1
                    end0 = int(e) - 1
                    if end0 >= start0:
                        intervals.append(
                            dict(accession=acc, name=name,
                                 type=typ, start0=start0, end0=end0)
                        )

    # Deduplicate identical intervals
    uniq = {}
    for itv in intervals:
        key = (itv["accession"], itv["type"], itv["start0"], itv["end0"])
        uniq[key] = itv
    return list(uniq.values())


# ---------------------INTERPRO2GO HELPERS ---------------------#


def load_interpro2go(file_path: str) -> dict[str, list[str]]:
    """
    Parses InterPro2GO file with lines like:

    InterPro:IPR000003 Retinoid X receptor/HNF4 > GO:DNA binding ; GO:0003677

    Args:
        file_path (str): Path to InterPro2GO file.

    Returns:
        { "IPR000003": ["GO:0003677"] }
    """
    interpro2go = defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()

            # skip comments / empty lines
            if not line or line.startswith("!"):
                continue

            # must contain InterPro and GO
            if not line.startswith("InterPro:") or ">" not in line:
                continue

            # extract InterPro accession
            try:
                ipr_part = line.split()[0]          # InterPro:IPR000003
                ipr = ipr_part.replace("InterPro:", "")
            except Exception:
                continue

            # extract all GO IDs on the line
            go_terms = GO_ID_REGEX.findall(line)

            for go in go_terms:
                interpro2go[ipr].append(go)

    return dict(interpro2go)


def load_molecular_function_go_ids(go_obo_path: str) -> set[str]:
    """
    Returns a set of GO IDs belonging to the Molecular Function namespace.
    """
    mf_go_ids = set()

    with open(go_obo_path, "r") as f:
        current_id = None
        current_namespace = None

        for line in f:
            line = line.strip()

            if line == "[Term]":
                current_id = None
                current_namespace = None

            elif line.startswith("id: GO:"):
                current_id = line.split("id: ")[1]

            elif line.startswith("namespace:"):
                current_namespace = line.split("namespace: ")[1]

            if current_id and current_namespace == "molecular_function":
                mf_go_ids.add(current_id)

    return mf_go_ids


def map_interpro_to_go(interpro_proto_matches, interpro2go, mf_go_ids):
    """
    Attaches a list of GO terms to each InterPro annotation.

    Args:
        interpro_proto_matches (list of dicts): Output of match_interpro_to_best_proto
        interpro2go (dict): Mapping of InterPro accession -> list of GO terms

    Returns:
        list of dicts:
        {
            "ipr_accession": str,
            "ipr_name": str,
            "ipr_type": str,
            "start0": int,
            "end0": int,
            "best_proto": int,
            "best_iou": float,
            "go_terms": list[str]
        }
    """

    mapped = []

    for row in interpro_proto_matches:
        ipr = row["ipr_accession"]

        go_terms = interpro2go.get(ipr)

        if not go_terms:
            continue  # skip InterPro entries without GO mapping

        # filter to molecular function
        mf_terms = [go for go in go_terms if go in mf_go_ids]

        if not mf_terms:
            continue

        mapped.append({
            **row,
            "go_terms": sorted(set(mf_terms)),
        })

    return mapped


# ---------------------INTERPRO AND PROTOTYPE MATCHING ---------------------#


def interval_iou(interval_a: set[int], interval_b: set[int]) -> float:
    """
    Intersection over Union between two residue index sets.
    Args:
        interval_a (set[int]): Set of residue indices for interval A.
        interval_b (set[int]): Set of residue indices for interval B.
    Returns:
        float: IoU value between 0.0 and 1.0
    """
    if not interval_a or not interval_b:
        return 0.0
    inter = len(interval_a & interval_b)
    union = len(interval_a | interval_b)
    return inter / union if union > 0 else 0.0


def match_interpro_to_best_proto(
    interpro_json: dict,
    segment_assignments_with_protos,
    pdb_id_full: str
):
    """
    For each InterPro annotation on a PDB chain:
      - compute IoU with each prototype
      - return best prototype assignment

    Args:
        interpro_json (dict): InterPro annotations JSON data.
        segment_assignments_with_protos (pd.DataFrame): DataFrame with segment assignments and prototype mappings.
        pdb_id (str): PDB identifier (with chain, e.g., "4K95-A_A").
        chain (str | None): Chain identifier to filter InterPro annotations (if None, no filtering).

    Returns:
        list of dicts, one per InterPro annotation:
        {
            "ipr_accession": str,
            "ipr_name": str,
            "ipr_type": str,
            "start0": int,
            "end0": int,
            "best_proto": int | None,
            "best_iou": float
        }
    """

    pdb4, chain = parse_pdb_chain(pdb_id_full)

    # --- extract interpro intervals ---
    interpro_intervals = extract_interpro_intervals(
        interpro_json=interpro_json,
        pdb4=pdb4,
        chain=chain,
    )

    if not interpro_intervals:
        return []

    # --- get residue → proto assignments ---
    residue_ids = get_residue_ids(segment_assignments_with_protos, pdb_id_full)
    proto_ids = get_proto_ids(segment_assignments_with_protos, pdb_id_full)

    # map proto -> residue set
    proto_to_residues = defaultdict(set)
    for res_id, proto_id in zip(residue_ids, proto_ids):
        proto_to_residues[proto_id].add(res_id)

    results = []

    for ipr in interpro_intervals:
        ipr_residues = set(range(ipr["start0"], ipr["end0"] + 1))

        best_proto = None
        best_iou = 0.0

        for proto_id, proto_residues in proto_to_residues.items():
            iou = interval_iou(ipr_residues, proto_residues)
            if iou > best_iou:
                best_iou = iou
                best_proto = proto_id

        results.append({
            "ipr_accession": ipr["accession"],
            "ipr_name": ipr["name"],
            "ipr_type": ipr["type"],
            "start0": ipr["start0"],
            "end0": ipr["end0"],
            "best_proto": best_proto,
            "best_iou": best_iou,
        })

    return results


# ---------------------PROTOTYPE 2 GO TERM MAP ---------------------#


def filter_enriched_go_terms_per_proto(df, proto_id):
    """Returns all enriched GO terms for a given prototype based on q-value.
    Args:
        df (pd.DataFrame): DataFrame with columns ['proto', 'go_term', 'qval', ...]
        proto_id (int): Prototype ID to filter by.
    Returns:
        pd.DataFrame: DataFrame of enriched GO terms for the given prototype, sorted by q-value.
    """
    proto_df = df[df['proto'] == proto_id]
    proto_df_significant = proto_df[proto_df['qval'] < 0.05]
    return proto_df_significant.sort_values('qval')


def attach_enriched_go_terms(
    interpro_go_mapped,
    enrichment_df,
):
    """
    For each InterPro annotation, attach enriched GO terms
    for its best prototype.

    Args:
        interpro_go_mapped (list of dicts): Output of map_interpro_to_go
        enrichment_df (pd.DataFrame): DataFrame with GO enrichment results,
            must contain columns ['proto', 'go_term', 'qval', ...]

    Returns:
        list of dicts:
        {
            ... (existing fields)
            "enriched_go_terms": list of dicts [
                {
                    "go_term": str,
                    "qval": float
                },
                ...
            ]
        }
    """

    results = []

    for row in interpro_go_mapped:
        proto_id = row["best_proto"]

        # get enriched GO terms for this prototype
        enriched_df = filter_enriched_go_terms_per_proto(
            enrichment_df,
            proto_id,
        )

        if enriched_df is None or enriched_df.empty:
            enriched_go = []
        else:
            enriched_go = (
                enriched_df
                .sort_values("qval", ascending=True)
                [["go_term", "qval"]]
                .to_dict("records")
            )

        results.append({
            **row,
            "enriched_go_terms": enriched_go,
        })

    return results


# ---------------------CALCULATE METRICS ---------------------#


def compute_ranking_metrics(true_go_terms, ranked_go_terms, ks=(1, 3, 5, 10)):
    """
    Args:
        true_go_terms: set[str]
        ranked_go_terms: list[str]
    Returns:
        dict with hit@k, precision@k, recall@k, mrr
    """

    true_set = set(true_go_terms)
    pred = ranked_go_terms

    metrics = {}

    # Hit / Precision / Recall
    for k in ks:
        topk = pred[:k]
        hits = len(true_set & set(topk))

        metrics[f"hit@{k}"] = 1 if hits > 0 else 0
        metrics[f"precision@{k}"] = hits / k
        metrics[f"recall@{k}"] = hits / len(true_set) if true_set else 0.0

    # MRR
    mrr = 0.0
    for i, go in enumerate(pred, start=1):
        if go in true_set:
            mrr = 1.0 / i
            break
    metrics["MRR"] = mrr

    return metrics


def evaluate_interpro_go_enrichment(attached_rows):
    """
    Computes ranking metrics for each InterPro annotation.

    Returns:
        list of dicts with metrics added
    """

    evaluated = []

    for row in attached_rows:
        ranked_go_terms = [
            x["go_term"] for x in row["enriched_go_terms"]
        ]

        metrics = compute_ranking_metrics(
            true_go_terms=row["go_terms"],
            ranked_go_terms=ranked_go_terms,
        )

        evaluated.append({
            **row,
            **metrics,
        })

    return evaluated


def summarize_metrics(evaluated_rows):
    df = pd.DataFrame(evaluated_rows)

    metric_cols = [c for c in df.columns if "@" in c or c == "MRR"]

    summary = {
        "overall": df[metric_cols].mean().to_dict(),
        "by_interpro_type": (
            df.groupby("ipr_type")[metric_cols]
            .mean()
            .to_dict(orient="index")
        )
    }

    return summary


# ---------------------MAIN DRIVER ---------------------#
def aggregate_overall_metrics(per_pdb_summaries):
    overall_rows = []

    for pdb_id, summary in per_pdb_summaries.items():
        if "overall" in summary:
            overall_rows.append(summary["overall"])

    if not overall_rows:
        return {}

    df = pd.DataFrame(overall_rows)
    return df.mean().to_dict()


def aggregate_by_interpro_type(per_pdb_summaries):
    from collections import defaultdict

    # type -> list of metric dicts
    type_to_rows = defaultdict(list)

    for pdb_id, summary in per_pdb_summaries.items():
        for ipr_type, metrics in summary.get("by_interpro_type", {}).items():
            type_to_rows[ipr_type].append(metrics)

    # compute mean per type
    aggregated = {}
    for ipr_type, rows in type_to_rows.items():
        df = pd.DataFrame(rows)
        aggregated[ipr_type] = df.mean().to_dict()

    return aggregated


def run_evaluation_over_all_pdbs(
    interpro_json,
    segment_assignments,
    prototype_assignments,
    enrichment_df,
    interpro2go,
    mf_go_ids,
    verbose=False
):
    """
    Runs InterPro → Proto → GO evaluation over all PDBs
    and aggregates metrics by averaging per-PDB summaries.

    Returns:
        {
            "per_pdb": dict[pdb_id -> summary],
            "mean_over_pdbs": dict[str -> float]
        }
    """

    segment_assignments_with_protos = add_residue_to_prototype_mapping(
        segment_assignments, prototype_assignments)

    pdb_ids = sorted(segment_assignments_with_protos["pdb_id"].unique())

    per_pdb_summaries = {}
    metric_keys = None

    for pdb_id in pdb_ids:

        # Step 1
        interpro_proto_matches = match_interpro_to_best_proto(
            interpro_json=interpro_json,
            segment_assignments_with_protos=segment_assignments_with_protos,
            pdb_id_full=pdb_id,
        )

        if not interpro_proto_matches:
            if verbose:
                print(f"No interpro annotations were found for {pdb_id}")
            continue

        # Step 2
        interpro_proto_matches = map_interpro_to_go(
            interpro_proto_matches,
            interpro2go,
            mf_go_ids
        )

        if not interpro_proto_matches:
            if verbose:
                print(f"No interpro2go matches were found for {pdb_id}")
            continue

        # Step 3
        interpro_proto_matches = attach_enriched_go_terms(
            interpro_proto_matches,
            enrichment_df,
        )

        # Step 4
        evaluated = evaluate_interpro_go_enrichment(
            interpro_proto_matches
        )

        if not evaluated:
            if verbose:
                print(f"Error evaluating {pdb_id}")
            continue

        summary = summarize_metrics(evaluated)
        per_pdb_summaries[pdb_id] = summary

        if metric_keys is None:
            metric_keys = list(summary.keys())

    mean_over_pdbs = {
        "overall": aggregate_overall_metrics(per_pdb_summaries),
        "by_interpro_type": aggregate_by_interpro_type(per_pdb_summaries),
    }

    return {
        "per_pdb": per_pdb_summaries,
        "mean_over_pdbs": mean_over_pdbs,
    }
