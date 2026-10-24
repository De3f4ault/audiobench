"""Community detection and Schools of Thought service module."""

from audiobench.community.community_service import (
    MACRO_COMMUNITY_MIN_SIZE,
    RECONCILIATION_JACCARD_THRESHOLD,
    assign_incremental,
    get_active_school_ids,
    get_community_summary,
    get_cross_school_bridge_pairs,
    get_related_schools,
    get_school_details,
    list_active_schools,
    run_canonical_clustering,
)
from audiobench.community.compute_medoids import (
    compute_soft_weighted_medoid,
    format_sample_passage,
    is_clean_candidate_chunk,
    populate_medoids,
    select_representative_samples,
)
from audiobench.community.naming import (
    generate_fallback_name,
    name_community,
    run_naming_sweep,
)

__all__ = [
    "run_canonical_clustering",
    "assign_incremental",
    "get_community_summary",
    "get_school_details",
    "list_active_schools",
    "get_active_school_ids",
    "get_related_schools",
    "get_cross_school_bridge_pairs",
    "populate_medoids",
    "compute_soft_weighted_medoid",
    "select_representative_samples",
    "format_sample_passage",
    "is_clean_candidate_chunk",
    "generate_fallback_name",
    "name_community",
    "run_naming_sweep",
    "MACRO_COMMUNITY_MIN_SIZE",
    "RECONCILIATION_JACCARD_THRESHOLD",
]


