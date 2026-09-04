"""schema.py -- the charter's probabilistic four-dimensional gravitational
scene graph.

    "The fundamental data object should not be a spreadsheet row.  It should be
     a probabilistic four-dimensional gravitational scene graph."

Node types, edge types and field types are EXACTLY the charter's lists, in the
charter's order, with no additions and no omissions -- `NODE_TYPES`,
`EDGE_TYPES` and `FIELD_TYPES` below are checked against the charter text by
`test_scene.py`.

The three structural rules this module enforces, at construction time:

  R1  Every node/edge attribute name must be registered in the metadata
      registry.  An unregistered quantity cannot enter a scene.  This is what
      makes the metadata contract binding rather than advisory.

  R2  A scene is a POSTERIOR.  Attribute values are `Fixed` or `Uncertain`;
      `Uncertain` carries a sampler, not an error bar, so that a correlated,
      bounded or multi-modal uncertainty survives.  `SceneGraph.realise()`
      draws one `SceneRealisation`; nothing downstream may read an `Uncertain`
      value directly.  `SceneRealisation` is a frozen snapshot, so a law cannot
      accidentally consume a mean.

  R3  Provenance is mandatory.  Every node names the observation or the
      generative assumption it came from, and whether that source presupposes
      dark matter.  `SceneGraph.dm_contaminated()` lists any node that does.
      The charter: a GR-derived convergence map or an NFW-fitted mass "is NOT a
      raw observation".

NO OBSERVATIONAL DATA IS OPENED BY THIS MODULE.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

import numpy as np

from metadata import (Dim, Quantity, Registry, ContractError, DIMLESS)

# ================================================================ vocabulary
# Exactly the charter's "Nodes" list, in the charter's order.
NODE_TYPES = (
    "star_population",        # stars or stellar tracer populations
    "galaxy",
    "gas_cell",               # gas cells or voxels
    "central_galaxy",
    "intracluster_light",
    "black_hole",
    "compact_substructure",
    "background_source",      # background lensed sources
    "observer",
    "instrument",
    "void",                   # -- the charter groups these four as one bullet,
    "filament",               #    "Voids, filaments, saddles, and boundaries".
    "saddle",                 #    They are separate node types here because
    "boundary",               #    each has a different support and a different
                              #    role in the ontology's direction section.
    "latent_field_cell",      # latent field cells in a candidate universe
)

#: the charter's own grouping, kept so the test can check coverage both ways
CHARTER_NODE_BULLETS = (
    "Stars or stellar tracer populations", "Galaxies", "Gas cells or voxels",
    "Central galaxies", "Intracluster light", "Black holes",
    "Compact substructures", "Background lensed sources", "Observer",
    "Instrument", "Voids, filaments, saddles, and boundaries",
    "Latent field cells in a candidate universe",
)

# Exactly the charter's "Edges" list, in the charter's order.
EDGE_TYPES = (
    "spatial_separation",
    "relative_velocity",
    "membership",
    "light_path",             # source-observer light path
    "source_source",          # source-source connection
    "tidal_pair",             # pairwise tidal relationship
    "orbital",                # orbital relationship
    "image_family",           # lensing-image family
    "causal_retarded",        # causal or retarded relationship
    "shared_covariance",      # shared measurement covariance
)

# Exactly the charter's "Fields" list, in the charter's order.
FIELD_TYPES = (
    "matter_density",
    "energy_stress",
    "velocity",
    "temperature_pressure",
    "em_state",
    "candidate_gravitational_state",
    "candidate_vacuum_state",
    "measurement_selection",
)

#: Which node types are SOURCES of gravity (they carry mass/energy) as opposed
#: to probes, apparatus or annotations.  A candidate law may only draw source
#: terms from these.
SOURCE_NODES = frozenset({
    "star_population", "galaxy", "gas_cell", "central_galaxy",
    "intracluster_light", "black_hole", "compact_substructure",
    "latent_field_cell",
})

#: Probe nodes: their observed behaviour is what a law must PREDICT.  Scoring a
#: law on a probe node is legitimate; using it as a source is double counting.
PROBE_NODES = frozenset({"background_source", "star_population", "galaxy",
                         "gas_cell"})

#: Annotation nodes: geometry of the scene, never sources.
ANNOTATION_NODES = frozenset({"void", "filament", "saddle", "boundary",
                              "observer", "instrument"})


# ============================================================ value wrappers

class Value:
    """Base for a scene attribute value."""
    def sample(self, rng: np.random.Generator):
        raise NotImplementedError

    def is_uncertain(self) -> bool:
        raise NotImplementedError


@dataclass
class Fixed(Value):
    """A value known to a precision the scene declares negligible.

    `why` must say WHY it is treated as exact, so that a silently-frozen
    uncertain quantity is visible in an audit.
    """
    v: Any
    why: str = "measured to negligible error at this scene's precision"

    def sample(self, rng):
        return self.v

    def is_uncertain(self):
        return False


@dataclass
class Uncertain(Value):
    """A posterior, represented by a sampler.

    Not a mean and a sigma.  The charter's requirement is an ENSEMBLE
    "consistent with redshifts, cluster phase space, morphology, substructure,
    spatial selection", and those constraints produce bounded, skewed and
    multi-modal posteriors that a Gaussian summary destroys.

    `draw(rng, n)` returns n draws; `summary` is for reporting only and is
    never read by a law.
    """
    draw: Callable[[np.random.Generator, int], np.ndarray]
    support: Tuple[float, float] = (-math.inf, math.inf)
    label: str = ""
    #: names of other attributes this value is correlated with.  A joint draw
    #: must be made through `SceneGraph.realise`, which honours these.
    correlated_with: Tuple[str, ...] = ()

    def sample(self, rng):
        return float(np.atleast_1d(self.draw(rng, 1))[0])

    def is_uncertain(self):
        return True

    def summary(self, rng: np.random.Generator, n: int = 4096) -> Dict[str, float]:
        x = np.asarray(self.draw(rng, n), dtype=float).ravel()
        return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                "p16": float(np.percentile(x, 16)),
                "p50": float(np.percentile(x, 50)),
                "p84": float(np.percentile(x, 84)),
                "min": float(x.min()), "max": float(x.max())}


# =================================================================== objects

@dataclass
class Node:
    id: str
    node_type: str
    attrs: Dict[str, Value] = field(default_factory=dict)
    #: provenance: what observation or assumption produced this node
    source: str = "unspecified"
    #: does the source of this node presuppose dark matter?  R3.
    presupposes_dm: bool = False
    dm_reason: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.node_type not in NODE_TYPES:
            raise ContractError(f"node {self.id}: unknown type "
                                f"{self.node_type!r}")


@dataclass
class Edge:
    id: str
    edge_type: str
    src: str
    dst: str
    attrs: Dict[str, Value] = field(default_factory=dict)
    source: str = "unspecified"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.edge_type not in EDGE_TYPES:
            raise ContractError(f"edge {self.id}: unknown type "
                                f"{self.edge_type!r}")


@dataclass
class FieldBlock:
    """A field sampled on a declared support with a declared smoothing scale.

    A field with no stated smoothing scale is not a physical object -- the
    charter puts "Resolution scale" and "Smoothing scale" in the ontology as
    candidate PHYSICAL variables, so leaving them implicit hides a variable.
    """
    name: str
    field_type: str
    #: (n, 3) positions in metres, in the scene frame
    points: np.ndarray
    #: (n,) or (n, k) values, or a callable (rng) -> array for an uncertain one
    values: Any
    dim: Dim = DIMLESS
    smoothing_m: float = 0.0
    frame: str = "cluster_rest"
    source: str = "unspecified"
    presupposes_dm: bool = False
    dm_reason: str = ""

    def __post_init__(self):
        if self.field_type not in FIELD_TYPES:
            raise ContractError(f"field {self.name}: unknown type "
                                f"{self.field_type!r}")
        self.points = np.asarray(self.points, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ContractError(f"field {self.name}: points must be (n, 3), "
                                f"got {self.points.shape}")


# ============================================================== realisations

@dataclass(frozen=True)
class SceneRealisation:
    """One concrete 3-D scene drawn from the posterior.

    Frozen on purpose.  A candidate law evaluates on a realisation, and it must
    not be able to reach back and read a mean or an error bar -- that is the
    exact move the charter forbids ("Do not collapse uncertainty to a best-fit
    scene").
    """
    scene_id: str
    draw_index: int
    seed: int
    node_attrs: Dict[str, Dict[str, Any]]
    edge_attrs: Dict[str, Dict[str, Any]]
    fields: Dict[str, Any]
    log_weight: float = 0.0

    def positions(self, graph: "SceneGraph",
                  types: Optional[Iterable[str]] = None) -> np.ndarray:
        """(n, 3) positions of nodes of the given types, in metres."""
        want = None if types is None else set(types)
        out = []
        for nid, n in graph.nodes.items():
            if want is not None and n.node_type not in want:
                continue
            a = self.node_attrs.get(nid, {})
            if all(k in a for k in ("x", "y", "z")):
                out.append((a["x"], a["y"], a["z"]))
        return np.asarray(out, dtype=float).reshape(-1, 3)

    def masses(self, graph: "SceneGraph",
               types: Optional[Iterable[str]] = None) -> np.ndarray:
        want = None if types is None else set(types)
        out = []
        for nid, n in graph.nodes.items():
            if want is not None and n.node_type not in want:
                continue
            a = self.node_attrs.get(nid, {})
            if all(k in a for k in ("x", "y", "z")):
                out.append(a.get("mass", 0.0))
        return np.asarray(out, dtype=float)


@dataclass
class SceneEnsemble:
    """N realisations plus their log weights.  This IS the scene."""
    scene_id: str
    draws: List[SceneRealisation]
    seed: int
    generator: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        return len(self.draws)

    def weights(self) -> np.ndarray:
        lw = np.array([d.log_weight for d in self.draws], dtype=float)
        lw -= lw.max()
        w = np.exp(lw)
        return w / w.sum()

    def ess(self) -> float:
        """Effective sample size.  A collapsed ESS means the ensemble is
        secretly a point estimate, which is the failure mode this whole
        object exists to prevent."""
        w = self.weights()
        return float(1.0 / np.sum(w ** 2))

    def expectation(self, f: Callable[[SceneRealisation], float]
                    ) -> Tuple[float, float]:
        """Weighted mean and sd of a functional OF THE SCENE.

        Note the order: f is applied to each realisation and THEN averaged.
        That is the charter's root-data rule at the level of the ensemble --
        E[f(scene)], never f(E[scene]).
        """
        v = np.array([f(d) for d in self.draws], dtype=float)
        w = self.weights()
        m = float(np.sum(w * v))
        var = float(np.sum(w * (v - m) ** 2))
        return m, math.sqrt(max(var, 0.0))


# ================================================================ SceneGraph

class SceneGraph:
    """A probabilistic 4-D gravitational scene.

    4-D: every node carries a time coordinate `t` (the epoch its state refers
    to) and light-path edges carry a retardation, so a candidate law with
    memory or finite propagation speed can ask for the source configuration on
    the past light cone rather than at the present instant.
    """

    def __init__(self, scene_id: str, registry: Registry,
                 frame: str = "cluster_rest", notes: str = ""):
        self.scene_id = scene_id
        self.registry = registry
        self.frame = frame
        self.notes = notes
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.fields: Dict[str, FieldBlock] = {}
        #: joint samplers: name -> (attr keys it produces, callable)
        self._joint: List[Tuple[Tuple[str, ...], Callable]] = []

    # --------------------------------------------------------------- R1
    def _check_attrs(self, owner: str, attrs: Dict[str, Value]):
        for k, v in attrs.items():
            if k not in self.registry:
                raise ContractError(
                    f"{owner}: attribute {k!r} is not in the metadata "
                    f"registry.  Every quantity entering a scene must carry "
                    f"the parameter metadata contract.")
            if not isinstance(v, Value):
                raise ContractError(
                    f"{owner}.{k}: value must be Fixed or Uncertain, got "
                    f"{type(v).__name__}.  A bare number hides whether the "
                    f"quantity is known or sampled.")

    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ContractError(f"duplicate node id {node.id!r}")
        self._check_attrs(f"node {node.id}", node.attrs)
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        if edge.id in self.edges:
            raise ContractError(f"duplicate edge id {edge.id!r}")
        for end in (edge.src, edge.dst):
            if end not in self.nodes:
                raise ContractError(f"edge {edge.id}: endpoint {end!r} "
                                    f"is not a node in this scene")
        self._check_attrs(f"edge {edge.id}", edge.attrs)
        self.edges[edge.id] = edge
        return edge

    def add_field(self, blk: FieldBlock) -> FieldBlock:
        if blk.name in self.fields:
            raise ContractError(f"duplicate field {blk.name!r}")
        self.fields[blk.name] = blk
        return blk

    def add_joint_sampler(self, keys: Sequence[str], fn: Callable):
        """Register a sampler that draws several correlated attributes at once.

        `fn(rng, graph) -> {node_id: {attr: value}}`.  Used for line-of-sight
        depth, which is correlated across members through the cluster's phase
        space and cannot be drawn per-node independently.
        """
        self._joint.append((tuple(keys), fn))

    # --------------------------------------------------------------- R3
    def dm_contaminated(self) -> List[Dict[str, str]]:
        out = []
        for nid, n in self.nodes.items():
            if n.presupposes_dm:
                out.append({"kind": "node", "id": nid,
                            "source": n.source, "reason": n.dm_reason})
        for fn, f in self.fields.items():
            if f.presupposes_dm:
                out.append({"kind": "field", "id": fn,
                            "source": f.source, "reason": f.dm_reason})
        return out

    # --------------------------------------------------------------- R2
    def realise(self, seed: int, draw_index: int = 0) -> SceneRealisation:
        rng = np.random.default_rng([seed, draw_index])
        node_attrs: Dict[str, Dict[str, Any]] = {}
        edge_attrs: Dict[str, Dict[str, Any]] = {}
        log_w = 0.0

        # joint samplers first: they set correlated attributes
        joint: Dict[str, Dict[str, Any]] = {}
        for keys, fn in self._joint:
            got = fn(rng, self)
            if isinstance(got, tuple):
                got, dlw = got
                log_w += float(dlw)
            for nid, kv in got.items():
                joint.setdefault(nid, {}).update(kv)

        for nid, n in self.nodes.items():
            a: Dict[str, Any] = {}
            for k, v in n.attrs.items():
                a[k] = v.sample(rng)
            a.update(joint.get(nid, {}))          # joint overrides marginal
            node_attrs[nid] = a
        for eid, e in self.edges.items():
            edge_attrs[eid] = {k: v.sample(rng) for k, v in e.attrs.items()}

        flds: Dict[str, Any] = {}
        for fn, f in self.fields.items():
            flds[fn] = (f.values(rng) if callable(f.values)
                        else np.asarray(f.values))

        return SceneRealisation(scene_id=self.scene_id, draw_index=draw_index,
                                seed=seed, node_attrs=node_attrs,
                                edge_attrs=edge_attrs, fields=flds,
                                log_weight=log_w)

    def ensemble(self, n: int, seed: int, generator: str = "") -> SceneEnsemble:
        draws = [self.realise(seed, i) for i in range(n)]
        return SceneEnsemble(scene_id=self.scene_id, draws=draws, seed=seed,
                             generator=generator or "SceneGraph.ensemble")

    # ------------------------------------------------------------ reporting
    def counts(self) -> Dict[str, int]:
        c = {f"node:{t}": 0 for t in NODE_TYPES}
        c.update({f"edge:{t}": 0 for t in EDGE_TYPES})
        c.update({f"field:{t}": 0 for t in FIELD_TYPES})
        for n in self.nodes.values():
            c[f"node:{n.node_type}"] += 1
        for e in self.edges.values():
            c[f"edge:{e.edge_type}"] += 1
        for f in self.fields.values():
            c[f"field:{f.field_type}"] += 1
        return c

    def uncertain_attrs(self) -> List[str]:
        out = []
        for nid, n in self.nodes.items():
            for k, v in n.attrs.items():
                if v.is_uncertain():
                    out.append(f"{nid}.{k}")
        return sorted(out)

    def fingerprint(self) -> str:
        """A stable hash of the STRUCTURE (not the draws).  Two scenes with
        the same fingerprint contain the same objects and relationships."""
        h = hashlib.sha256()
        h.update(self.scene_id.encode())
        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            h.update(f"{nid}|{n.node_type}|{sorted(n.attrs)}".encode())
        for eid in sorted(self.edges):
            e = self.edges[eid]
            h.update(f"{eid}|{e.edge_type}|{e.src}|{e.dst}".encode())
        for fn in sorted(self.fields):
            f = self.fields[fn]
            h.update(f"{fn}|{f.field_type}|{f.points.shape}".encode())
        return h.hexdigest()[:16]

    def to_json(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "frame": self.frame,
            "fingerprint": self.fingerprint(),
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_fields": len(self.fields),
            "counts": self.counts(),
            "n_uncertain_attrs": len(self.uncertain_attrs()),
            "dm_contaminated": self.dm_contaminated(),
            "notes": self.notes,
        }
