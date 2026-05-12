"""
EntityResolver – GPU-accelerated embedding-based cross-article entity
resolution using SentenceTransformers (BAAI/bge-large-en-v1.5).

Workflow:
  1. Encode all unique entity names + LLM-provided aliases with BGE-large
  2. Build a cosine-similarity matrix on GPU
  3. Agglomerative clustering with a conservative threshold (default 0.82)
  4. Within each cluster: pick canonical name, merge types/aliases/sources
  5. Output: mapping from every raw entity string → CanonicalEntity
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CanonicalEntity:
    """A resolved canonical entity that may aggregate multiple surface forms."""
    canonical_name: str
    entity_type: str = "Other"
    normalized_id: str = "N/A"
    aliases: List[str] = field(default_factory=list)
    source_articles: List[str] = field(default_factory=list)
    mention_count: int = 0


class EntityResolver:
    """
    Embedding-based entity resolution with GPU acceleration.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model name (default: ``BAAI/bge-large-en-v1.5``).
    similarity_threshold : float
        Cosine similarity threshold for clustering (default: 0.82).
        Higher = more conservative (fewer merges).
    device : str
        PyTorch device (default: ``cuda`` if available, else ``cpu``).
    batch_size : int
        Encoding batch size for the embedding model.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        similarity_threshold: float = 0.90,
        device: Optional[str] = None,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(
            "Initializing EntityResolver with %s on %s (threshold=%.2f)",
            model_name,
            self.device,
            similarity_threshold,
        )

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=self.device)

    # ── public API ──────────────────────────────────────────────────────
    def resolve(
        self,
        typed_entities: Dict[str, "TypedEntity"],  # from entity_typer
    ) -> Tuple[Dict[str, CanonicalEntity], Dict[str, str]]:
        """
        Resolve entities across articles into canonical forms.

        Strategy:
          1. Partition entities by LLM-assigned type (only merge within
             the same type to prevent cross-category chaining).
          2. Within each type partition, encode + cluster with complete
             linkage at the configured similarity threshold.
          3. Also apply alias-based merging: if entity A has alias "X"
             and entity B is named "X", they should merge regardless
             of embedding distance.

        Parameters
        ----------
        typed_entities : dict[str, TypedEntity]
            Output from EntityTyper.classify_entities().

        Returns
        -------
        canonical_entities : dict[str, CanonicalEntity]
            Mapping canonical_name → CanonicalEntity.
        name_mapping : dict[str, str]
            Mapping every raw entity string → canonical_name.
        """
        if not typed_entities:
            return {}, {}

        raw_names = sorted(typed_entities.keys())

        # ── Step 0: Alias-based pre-merging (union-find on aliases) ──────
        # If entity A lists "X" as an alias and entity B is called "X",
        # they refer to the same concept regardless of embeddings.
        alias_parent: Dict[str, str] = {}  # raw_name → representative

        # Build lower-cased lookup: lowered_name → [raw_names]
        name_to_raws: Dict[str, List[str]] = {}
        for raw in raw_names:
            name_to_raws.setdefault(raw.lower().strip(), []).append(raw)
        # Also index aliases → owner
        alias_to_raws: Dict[str, List[str]] = {}
        for raw in raw_names:
            te = typed_entities[raw]
            for alias in te.aliases:
                alias_to_raws.setdefault(alias.lower().strip(), []).append(raw)

        # Simple union-find for alias merging
        uf_parent: Dict[str, str] = {r: r for r in raw_names}

        def uf_find(x: str) -> str:
            while uf_parent[x] != x:
                uf_parent[x] = uf_parent[uf_parent[x]]
                x = uf_parent[x]
            return x

        def uf_union(a: str, b: str):
            ra, rb = uf_find(a), uf_find(b)
            if ra != rb:
                uf_parent[ra] = rb

        # Merge via alias overlap
        for raw in raw_names:
            te = typed_entities[raw]
            for alias in te.aliases:
                akey = alias.lower().strip()
                # If another raw entity has this name, merge them
                if akey in name_to_raws:
                    for other_raw in name_to_raws[akey]:
                        if other_raw != raw:
                            uf_union(raw, other_raw)
                # If another entity also claims this alias, merge
                if akey in alias_to_raws:
                    for other_raw in alias_to_raws[akey]:
                        if other_raw != raw:
                            uf_union(raw, other_raw)

        # Build alias-merge groups
        alias_groups: Dict[str, Set[str]] = {}
        for raw in raw_names:
            root = uf_find(raw)
            alias_groups.setdefault(root, set()).add(raw)

        logger.info(
            "Alias pre-merge: %d raw → %d groups",
            len(raw_names),
            len(alias_groups),
        )

        # For each alias group, pick a representative (most mentioned)
        # and assign a canonical type (most common non-Other type)
        group_representatives: Dict[str, str] = {}  # group_root → representative_name
        group_type: Dict[str, str] = {}  # group_root → entity_type
        for root, members in alias_groups.items():
            # Representative = most mentioned
            best = max(members, key=lambda r: typed_entities[r].mention_count)
            group_representatives[root] = best
            # Type = most common among members (prefer non-Other)
            type_counts: Dict[str, int] = {}
            for m in members:
                t = typed_entities[m].entity_type
                type_counts[t] = type_counts.get(t, 0) + typed_entities[m].mention_count
            # Prefer non-Other
            non_other = {t: c for t, c in type_counts.items() if t != "Other"}
            if non_other:
                group_type[root] = max(non_other, key=non_other.get)
            else:
                group_type[root] = "Other"

        # Now we have groups that are pre-merged by aliases.
        # Next: partition by type and apply embedding-based clustering.

        # ── Step 1: Partition by entity type ─────────────────────────────
        type_partitions: Dict[str, List[str]] = {}  # type → [group_roots]
        for root in alias_groups:
            etype = group_type[root]
            type_partitions.setdefault(etype, []).append(root)

        logger.info(
            "Type partitions: %s",
            {t: len(roots) for t, roots in type_partitions.items()},
        )

        # ── Step 2: Encode and cluster within each type partition ────────
        # Use the representative name for each group as the embedding target
        all_cluster_groups: List[Set[str]] = []  # final: list of sets of group_roots

        for etype, group_roots in type_partitions.items():
            if len(group_roots) <= 1:
                for root in group_roots:
                    all_cluster_groups.append({root})
                continue

            representatives = [group_representatives[root] for root in group_roots]

            # Encode representative names
            embeddings = self._encode(representatives)
            sim_matrix = self._cosine_similarity_matrix(embeddings)

            # Cluster with complete linkage
            sub_clusters = self._cluster(representatives, sim_matrix)

            for sc in sub_clusters:
                # Map back from representative names to group_roots
                merged_roots: Set[str] = set()
                for rep_name in sc:
                    # Find which group_root has this representative
                    for root in group_roots:
                        if group_representatives[root] == rep_name:
                            merged_roots.add(root)
                            break
                all_cluster_groups.append(merged_roots)

            logger.info(
                "  Type %s: %d groups → %d clusters",
                etype,
                len(group_roots),
                len([sc for sc in sub_clusters]),
            )

        # ── Step 3: Build canonical entities ─────────────────────────────
        canonical_entities: Dict[str, CanonicalEntity] = {}
        name_mapping: Dict[str, str] = {}

        for cluster_roots in all_cluster_groups:
            # Collect all raw entity names from all alias groups in this cluster
            raw_members: Set[str] = set()
            for root in cluster_roots:
                raw_members.update(alias_groups[root])

            # Pick canonical name: most frequent
            freq = Counter()
            all_aliases: Set[str] = set()
            all_sources: Set[str] = set()
            best_type = "Other"
            best_norm_id = "N/A"
            total_mentions = 0

            for raw in raw_members:
                te = typed_entities.get(raw)
                if te is None:
                    continue
                freq[raw] += te.mention_count
                all_aliases.add(raw)
                all_aliases.update(te.aliases)
                all_sources.update(te.source_articles)
                total_mentions += te.mention_count
                if te.entity_type != "Other" and best_type == "Other":
                    best_type = te.entity_type
                if te.normalized_id != "N/A" and best_norm_id == "N/A":
                    best_norm_id = te.normalized_id

            canonical_name = freq.most_common(1)[0][0] if freq else sorted(raw_members)[0]
            all_aliases.discard(canonical_name)

            ce = CanonicalEntity(
                canonical_name=canonical_name,
                entity_type=best_type,
                normalized_id=best_norm_id,
                aliases=sorted(all_aliases),
                source_articles=sorted(all_sources),
                mention_count=total_mentions,
            )
            canonical_entities[canonical_name] = ce

            for raw in raw_members:
                name_mapping[raw] = canonical_name

        # Ensure every original raw_name is mapped (even singletons)
        for raw in raw_names:
            if raw not in name_mapping:
                te = typed_entities[raw]
                ce = CanonicalEntity(
                    canonical_name=raw,
                    entity_type=te.entity_type,
                    normalized_id=te.normalized_id,
                    aliases=te.aliases,
                    source_articles=te.source_articles,
                    mention_count=te.mention_count,
                )
                canonical_entities[raw] = ce
                name_mapping[raw] = raw

        logger.info(
            "Entity resolution: %d raw → %d canonical (%.1f%% reduction)",
            len(raw_names),
            len(canonical_entities),
            (1 - len(canonical_entities) / max(len(raw_names), 1)) * 100,
        )
        return canonical_entities, name_mapping

    # ── internals ───────────────────────────────────────────────────────
    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode entity strings with the SentenceTransformer model."""
        # BGE models benefit from the instruction prefix for retrieval
        prefix = "Represent this biomedical entity for clustering: "
        prefixed = [f"{prefix}{t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine sim
        )
        return embeddings

    def _cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity on GPU (or CPU fallback).

        Since embeddings are L2-normalized, cosine sim = dot product.
        """
        n = len(embeddings)
        logger.info("Computing %d×%d similarity matrix", n, n)

        t = torch.from_numpy(embeddings).to(self.device)
        # Dot product of L2-normalized vectors = cosine similarity
        sim = torch.mm(t, t.T)
        result = sim.cpu().numpy()
        del t, sim
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return result

    def _cluster(
        self, surface_forms: List[str], sim_matrix: np.ndarray
    ) -> List[List[str]]:
        """
        Agglomerative clustering with **complete linkage** (farthest neighbor).

        Complete linkage requires ALL members in a cluster to be within the
        similarity threshold of each other, avoiding the "chain effect" that
        single-linkage (union-find) produces with dense embedding spaces.
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform

        n = len(surface_forms)
        if n <= 1:
            return [[sf] for sf in surface_forms]

        # Convert similarity matrix to distance matrix (1 - sim)
        dist_matrix = 1.0 - sim_matrix
        # Ensure diagonal is exactly 0 and matrix is symmetric
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.maximum(dist_matrix, 0.0)  # clip tiny negatives
        dist_matrix = (dist_matrix + dist_matrix.T) / 2  # force symmetry

        # Convert to condensed form for scipy
        condensed = squareform(dist_matrix, checks=False)

        # Complete linkage: distance between clusters = max distance
        # between any two members → conservative, avoids chaining
        Z = linkage(condensed, method="complete")

        # Cut at distance threshold = 1 - similarity_threshold
        distance_threshold = 1.0 - self.similarity_threshold
        labels = fcluster(Z, t=distance_threshold, criterion="distance")

        # Group by cluster label
        groups: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(i)

        clusters = [
            [surface_forms[i] for i in members] for members in groups.values()
        ]
        multi = [c for c in clusters if len(c) > 1]

        # Log some example multi-member clusters for debugging
        for c in sorted(multi, key=len, reverse=True)[:10]:
            logger.info("  Cluster (%d members): %s", len(c), c[:5])

        logger.info(
            "Clustering: %d clusters (%d multi-member, %d singletons)",
            len(clusters),
            len(multi),
            len(clusters) - len(multi),
        )
        return clusters
