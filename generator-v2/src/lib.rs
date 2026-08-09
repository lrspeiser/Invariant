use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GeneratorConfig {
    pub protocol_version: String,
    pub scope_claim: String,
    pub basis_count: usize,
    pub max_action_terms: usize,
    pub coefficient_alphabet: Vec<i8>,
    pub shared_coupling: String,
    pub coupling_magnitude: f64,
    pub convexity_tolerance: f64,
    pub convexity_samples: ConvexitySamples,
    pub maximum_universal_constants: usize,
    pub universal_constants: Vec<String>,
    pub sample_limit: usize,
    pub observational_data_opened: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConvexitySamples {
    pub d: Vec<f64>,
    pub p: Vec<f64>,
    pub state: Vec<f64>,
}

impl GeneratorConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.basis_count == 0 || self.basis_count > u16::MAX as usize {
            return Err("basis_count must be in 1..=65535".into());
        }
        if self.max_action_terms == 0 || self.max_action_terms > self.basis_count {
            return Err("max_action_terms must be in 1..=basis_count".into());
        }
        if self.max_action_terms > 20 {
            return Err("max_action_terms above 20 is outside the v2 ordinal encoding".into());
        }
        if self.coefficient_alphabet != vec![-1, 1] {
            return Err("v2 requires the frozen coefficient alphabet [-1, 1]".into());
        }
        if !self.coupling_magnitude.is_finite() || self.coupling_magnitude <= 0.0 {
            return Err("coupling_magnitude must be finite and positive".into());
        }
        if !self.convexity_tolerance.is_finite() || self.convexity_tolerance < 0.0 {
            return Err("convexity_tolerance must be finite and non-negative".into());
        }
        if self.convexity_samples.d.is_empty()
            || self.convexity_samples.p.is_empty()
            || self.convexity_samples.state.is_empty()
        {
            return Err("convexity sample axes may not be empty".into());
        }
        if self.universal_constants.len() > self.maximum_universal_constants {
            return Err("universal constant cap exceeded".into());
        }
        if self.observational_data_opened {
            return Err("Generator tiers 0-2 may not open observational data".into());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Transform {
    Identity,
    Sqrt1pMinus1,
    Saturate,
}

#[derive(Clone, Debug, Serialize)]
pub struct BasisTerm {
    pub id: u16,
    pub px: u8,
    pub pq: u8,
    pub pz: u8,
    pub transform: Transform,
    pub dimension_l: i8,
    pub dimension_t: i8,
    pub derivative_order_in_h: u8,
    pub has_measured_state: bool,
    pub high_field_growth_numerator: u8,
    pub high_field_growth_denominator: u8,
    pub expression: String,
}

impl BasisTerm {
    fn monomial(px: u8, pq: u8, pz: u8) -> String {
        let mut factors = Vec::new();
        for (name, power) in [("x", px), ("q", pq), ("z", pz)] {
            match power {
                0 => {}
                1 => factors.push(name.to_string()),
                _ => factors.push(format!("{name}**{power}")),
            }
        }
        if factors.is_empty() {
            "1".into()
        } else {
            factors.join("*")
        }
    }

    fn new(id: u16, px: u8, pq: u8, pz: u8, transform: Transform) -> Self {
        let monomial = Self::monomial(px, pq, pz);
        let (growth_num, growth_den) = match transform {
            Transform::Identity => (px, 1),
            Transform::Sqrt1pMinus1 => (px, 2),
            Transform::Saturate => (0, 1),
        };
        let expression = match transform {
            Transform::Identity => monomial,
            Transform::Sqrt1pMinus1 => format!("sqrt(1+({monomial}))-1"),
            Transform::Saturate => format!("({monomial})/(1+({monomial}))"),
        };
        Self {
            id,
            px,
            pq,
            pz,
            transform,
            dimension_l: 0,
            dimension_t: 0,
            derivative_order_in_h: if pq > 0 { 1 } else { 0 },
            has_measured_state: pq > 0 || pz > 0,
            high_field_growth_numerator: growth_num,
            high_field_growth_denominator: growth_den,
            expression,
        }
    }
}

pub fn build_basis(count: usize) -> Vec<BasisTerm> {
    let transforms = [
        Transform::Identity,
        Transform::Sqrt1pMinus1,
        Transform::Saturate,
    ];
    let mut terms = Vec::with_capacity(count);
    'degree: for degree in 1u8..=20 {
        for px in 0..=degree {
            for pq in 0..=(degree - px) {
                let pz = degree - px - pq;
                for transform in transforms {
                    let id = terms.len() as u16;
                    terms.push(BasisTerm::new(id, px, pq, pz, transform));
                    if terms.len() == count {
                        break 'degree;
                    }
                }
            }
        }
    }
    assert_eq!(
        terms.len(),
        count,
        "requested basis exceeds generated library"
    );
    terms
}

pub fn choose(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for index in 0..k {
        result = result * (n - index) as u128 / (index + 1) as u128;
    }
    u64::try_from(result).expect("combination count exceeds u64")
}

pub fn family_count(basis_count: usize, terms: usize) -> u64 {
    choose(basis_count, terms)
        .checked_mul(1u64 << terms)
        .expect("family count exceeds u64")
}

pub fn total_search_count(config: &GeneratorConfig) -> u64 {
    (1..=config.max_action_terms)
        .map(|terms| family_count(config.basis_count, terms))
        .sum()
}

pub fn unrank_combination(n: usize, k: usize, mut rank: u64) -> Vec<u16> {
    assert!(rank < choose(n, k));
    let mut result = Vec::with_capacity(k);
    let mut start = 0usize;
    for position in 0..k {
        let remaining = k - position - 1;
        for value in start..n {
            let count = if remaining == 0 {
                1
            } else {
                choose(n - value - 1, remaining)
            };
            if rank < count {
                result.push(value as u16);
                start = value + 1;
                break;
            }
            rank -= count;
        }
    }
    result
}

#[derive(Clone, Debug)]
pub struct DecodedCandidate {
    pub ordinal: u64,
    pub term_ids: Vec<u16>,
    pub sign_mask: u32,
}

pub fn decode_ordinal(config: &GeneratorConfig, ordinal: u64) -> DecodedCandidate {
    let mut offset = 0u64;
    for terms in 1..=config.max_action_terms {
        let width = family_count(config.basis_count, terms);
        if ordinal < offset + width {
            let local = ordinal - offset;
            let sign_mask = (local & ((1u64 << terms) - 1)) as u32;
            let combination_rank = local >> terms;
            return DecodedCandidate {
                ordinal,
                term_ids: unrank_combination(config.basis_count, terms, combination_rank),
                sign_mask,
            };
        }
        offset += width;
    }
    panic!("ordinal {ordinal} outside search space of {offset}");
}

fn sign_at(sign_mask: u32, position: usize) -> i8 {
    if sign_mask & (1u32 << position) == 0 {
        -1
    } else {
        1
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GateCode {
    RejectFluxOnly,
    RejectHighField,
    RejectNoGradientSector,
    RejectNegativeElasticity,
    RejectSampledStaticConvexity,
    SurviveSampledStatic,
}

impl GateCode {
    fn byte(self) -> u8 {
        match self {
            Self::RejectFluxOnly => 1,
            Self::RejectHighField => 2,
            Self::RejectNoGradientSector => 3,
            Self::RejectNegativeElasticity => 4,
            Self::RejectSampledStaticConvexity => 5,
            Self::SurviveSampledStatic => 6,
        }
    }
}

fn evaluate_structural(candidate: &DecodedCandidate, basis: &[BasisTerm]) -> GateCode {
    let terms: Vec<&BasisTerm> = candidate
        .term_ids
        .iter()
        .map(|id| &basis[*id as usize])
        .collect();
    if !terms.iter().any(|term| term.has_measured_state) {
        return GateCode::RejectFluxOnly;
    }
    let mut dangerous_growth_coefficients: BTreeMap<(u8, u8), i32> = BTreeMap::new();
    for (position, term) in terms.iter().enumerate() {
        let mut numerator = term.high_field_growth_numerator;
        let mut denominator = term.high_field_growth_denominator;
        let divisor = gcd_u8(numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
        if numerator >= denominator {
            *dangerous_growth_coefficients
                .entry((numerator, denominator))
                .or_insert(0) += sign_at(candidate.sign_mask, position) as i32;
        }
    }
    if dangerous_growth_coefficients
        .values()
        .any(|coefficient| *coefficient != 0)
    {
        return GateCode::RejectHighField;
    }
    let gradient_positions: Vec<usize> = terms
        .iter()
        .enumerate()
        .filter_map(|(position, term)| (term.pq > 0).then_some(position))
        .collect();
    if gradient_positions.is_empty() {
        return GateCode::RejectNoGradientSector;
    }
    if gradient_positions.len() == 1 {
        let position = gradient_positions[0];
        let term = terms[position];
        if term.transform == Transform::Identity
            && (term.px, term.pq, term.pz) == (0, 1, 0)
            && sign_at(candidate.sign_mask, position) < 0
        {
            return GateCode::RejectNegativeElasticity;
        }
    }
    GateCode::SurviveSampledStatic
}

fn gcd_u8(mut left: u8, mut right: u8) -> u8 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left.max(1)
}

#[derive(Clone, Copy, Debug)]
struct Jet2 {
    value: f64,
    gd: f64,
    gp: f64,
    hdd: f64,
    hdp: f64,
    hpp: f64,
}

impl Jet2 {
    fn constant(value: f64) -> Self {
        Self {
            value,
            gd: 0.0,
            gp: 0.0,
            hdd: 0.0,
            hdp: 0.0,
            hpp: 0.0,
        }
    }

    fn d(value: f64) -> Self {
        Self {
            value,
            gd: 1.0,
            gp: 0.0,
            hdd: 0.0,
            hdp: 0.0,
            hpp: 0.0,
        }
    }

    fn p(value: f64) -> Self {
        Self {
            value,
            gd: 0.0,
            gp: 1.0,
            hdd: 0.0,
            hdp: 0.0,
            hpp: 0.0,
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            gd: self.gd + other.gd,
            gp: self.gp + other.gp,
            hdd: self.hdd + other.hdd,
            hdp: self.hdp + other.hdp,
            hpp: self.hpp + other.hpp,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            value: self.value * other.value,
            gd: self.gd * other.value + self.value * other.gd,
            gp: self.gp * other.value + self.value * other.gp,
            hdd: self.hdd * other.value + self.value * other.hdd + 2.0 * self.gd * other.gd,
            hdp: self.hdp * other.value
                + self.value * other.hdp
                + self.gd * other.gp
                + self.gp * other.gd,
            hpp: self.hpp * other.value + self.value * other.hpp + 2.0 * self.gp * other.gp,
        }
    }

    fn unary(self, value: f64, first: f64, second: f64) -> Self {
        Self {
            value,
            gd: first * self.gd,
            gp: first * self.gp,
            hdd: first * self.hdd + second * self.gd * self.gd,
            hdp: first * self.hdp + second * self.gd * self.gp,
            hpp: first * self.hpp + second * self.gp * self.gp,
        }
    }

    fn sqrt(self) -> Self {
        let root = self.value.sqrt();
        self.unary(root, 0.5 / root, -0.25 / (self.value * root))
    }

    fn reciprocal(self) -> Self {
        self.unary(
            1.0 / self.value,
            -1.0 / self.value.powi(2),
            2.0 / self.value.powi(3),
        )
    }

    fn powu(self, power: u8) -> Self {
        let mut result = Self::constant(1.0);
        for _ in 0..power {
            result = result.mul(self);
        }
        result
    }
}

fn term_jet(term: &BasisTerm, d: f64, p: f64, state: f64) -> Jet2 {
    let x = Jet2::d(d).powu(2);
    let q = Jet2::p(p).powu(2);
    let z = Jet2::constant(state * state);
    let monomial = x.powu(term.px).mul(q.powu(term.pq)).mul(z.powu(term.pz));
    match term.transform {
        Transform::Identity => monomial,
        Transform::Sqrt1pMinus1 => monomial
            .add(Jet2::constant(1.0))
            .sqrt()
            .add(Jet2::constant(-1.0)),
        Transform::Saturate => monomial.mul(monomial.add(Jet2::constant(1.0)).reciprocal()),
    }
}

type HessianRow = [f64; 3];

fn precompute_hessians(config: &GeneratorConfig, basis: &[BasisTerm]) -> Vec<Vec<HessianRow>> {
    basis
        .iter()
        .map(|term| {
            let mut rows = Vec::new();
            for &d in &config.convexity_samples.d {
                for &p in &config.convexity_samples.p {
                    for &state in &config.convexity_samples.state {
                        let jet = term_jet(term, d, p, state);
                        rows.push([jet.hdd, jet.hdp, jet.hpp]);
                    }
                }
            }
            rows
        })
        .collect()
}

fn sampled_convexity_passes(
    candidate: &DecodedCandidate,
    config: &GeneratorConfig,
    hessians: &[Vec<HessianRow>],
) -> bool {
    let sample_rows = hessians.first().map(Vec::as_slice).unwrap_or(&[]);
    for (sample, _) in sample_rows.iter().enumerate() {
        let mut hdd = 1.0;
        let mut hdp = 0.0;
        let mut hpp = 0.0;
        for (position, &term_id) in candidate.term_ids.iter().enumerate() {
            let sign = sign_at(candidate.sign_mask, position) as f64;
            let row = hessians[term_id as usize][sample];
            let scale = config.coupling_magnitude * sign;
            hdd += scale * row[0];
            hdp += scale * row[1];
            hpp += scale * row[2];
        }
        let discriminant = ((hdd - hpp) * (hdd - hpp) + 4.0 * hdp * hdp).max(0.0);
        let minimum = 0.5 * (hdd + hpp - discriminant.sqrt());
        if !minimum.is_finite() || minimum <= config.convexity_tolerance {
            return false;
        }
    }
    true
}

pub fn evaluate_candidate(
    candidate: &DecodedCandidate,
    basis: &[BasisTerm],
    config: &GeneratorConfig,
    hessians: &[Vec<HessianRow>],
) -> GateCode {
    let structural = evaluate_structural(candidate, basis);
    if structural != GateCode::SurviveSampledStatic {
        return structural;
    }
    if !sampled_convexity_passes(candidate, config, hessians) {
        return GateCode::RejectSampledStaticConvexity;
    }
    GateCode::SurviveSampledStatic
}

pub fn candidate_digest(config: &GeneratorConfig, candidate: &DecodedCandidate) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"SIGMA-GENERATOR-V2\0");
    hasher.update(config.protocol_version.as_bytes());
    hasher.update([0]);
    hasher.update([candidate.term_ids.len() as u8]);
    for (position, term_id) in candidate.term_ids.iter().enumerate() {
        hasher.update(term_id.to_le_bytes());
        hasher.update([if sign_at(candidate.sign_mask, position) > 0 {
            1
        } else {
            0
        }]);
    }
    hasher.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").unwrap();
    }
    output
}

pub fn correction_expression(candidate: &DecodedCandidate, basis: &[BasisTerm]) -> String {
    candidate
        .term_ids
        .iter()
        .enumerate()
        .map(|(position, id)| {
            let sign = if sign_at(candidate.sign_mask, position) > 0 {
                "+"
            } else {
                "-"
            };
            format!("{sign}({})", basis[*id as usize].expression)
        })
        .collect::<Vec<_>>()
        .join("")
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CandidateSample {
    pub candidate_id: String,
    pub ordinal: u64,
    pub term_ids: Vec<u16>,
    pub signs: Vec<i8>,
    pub correction_expression: String,
    pub gate: GateCode,
}

fn sample_candidate(
    config: &GeneratorConfig,
    candidate: &DecodedCandidate,
    basis: &[BasisTerm],
    gate: GateCode,
) -> CandidateSample {
    let digest = candidate_digest(config, candidate);
    CandidateSample {
        candidate_id: format!("STC2-{}", &hex(&digest)[..24]),
        ordinal: candidate.ordinal,
        term_ids: candidate.term_ids.clone(),
        signs: (0..candidate.term_ids.len())
            .map(|position| sign_at(candidate.sign_mask, position))
            .collect(),
        correction_expression: correction_expression(candidate, basis),
        gate,
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BlockManifest {
    pub block_index: u64,
    pub start_ordinal: u64,
    pub end_ordinal_exclusive: u64,
    pub processed: u64,
    pub survivors: u64,
    pub digest_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub survivor_export: Option<SurvivorBlockManifest>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SurvivorBlockManifest {
    pub file: String,
    pub record_format: String,
    pub record_count: u64,
    pub file_size_bytes: u64,
    pub file_sha256: String,
}

#[derive(Clone, Debug)]
struct CompactSurvivor {
    ordinal: u64,
    sign_mask: u8,
    term_ids: Vec<u16>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct BlockResult {
    manifest: BlockManifest,
    gate_counts: BTreeMap<GateCode, u64>,
    samples: Vec<CandidateSample>,
    #[serde(skip, default)]
    survivor_records: Vec<CompactSurvivor>,
}

fn process_block(
    block_index: u64,
    start: u64,
    end: u64,
    config: &GeneratorConfig,
    basis: &[BasisTerm],
    hessians: &[Vec<HessianRow>],
    collect_survivors: bool,
) -> BlockResult {
    let mut hasher = Sha256::new();
    hasher.update(b"SIGMA-GENERATOR-V2-BLOCK\0");
    hasher.update(block_index.to_le_bytes());
    hasher.update(start.to_le_bytes());
    hasher.update(end.to_le_bytes());
    let mut gate_counts = BTreeMap::new();
    let mut sample_buckets: BTreeMap<GateCode, Vec<([u8; 32], CandidateSample)>> = BTreeMap::new();
    let mut survivor_records = Vec::new();
    for ordinal in start..end {
        let candidate = decode_ordinal(config, ordinal);
        let gate = evaluate_candidate(&candidate, basis, config, hessians);
        *gate_counts.entry(gate).or_insert(0) += 1;
        let digest = candidate_digest(config, &candidate);
        hasher.update(digest);
        hasher.update([gate.byte()]);
        if collect_survivors && gate == GateCode::SurviveSampledStatic {
            survivor_records.push(CompactSurvivor {
                ordinal,
                sign_mask: candidate.sign_mask as u8,
                term_ids: candidate.term_ids.clone(),
            });
        }
        let bucket = sample_buckets.entry(gate).or_default();
        if bucket.len() < 4 || digest < bucket.last().unwrap().0 {
            bucket.push((digest, sample_candidate(config, &candidate, basis, gate)));
            bucket.sort_by_key(|item| item.0);
            bucket.truncate(4);
        }
    }
    let survivors = *gate_counts
        .get(&GateCode::SurviveSampledStatic)
        .unwrap_or(&0);
    BlockResult {
        manifest: BlockManifest {
            block_index,
            start_ordinal: start,
            end_ordinal_exclusive: end,
            processed: end - start,
            survivors,
            digest_sha256: hex(&hasher.finalize()),
            survivor_export: None,
        },
        gate_counts,
        samples: sample_buckets
            .into_values()
            .flat_map(|bucket| bucket.into_iter().map(|(_, sample)| sample))
            .collect(),
        survivor_records,
    }
}

#[derive(Clone, Debug)]
pub struct RunOptions {
    pub start_ordinal: u64,
    pub end_ordinal_exclusive: u64,
    pub block_size: u64,
    pub threads: usize,
    pub shard_index: u64,
    pub shard_count: u64,
    pub config_sha256: String,
    pub checkpoint_directory: Option<PathBuf>,
    pub survivor_directory: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunManifest {
    pub schema_version: String,
    pub generator_version: String,
    pub protocol_version: String,
    pub scope_claim: String,
    pub config_sha256: String,
    pub basis_count: usize,
    pub basis_library_sha256: String,
    pub max_action_terms: usize,
    pub coefficient_alphabet: Vec<i8>,
    pub coefficient_semantics: String,
    pub total_declared_actions: u64,
    pub shard_index: u64,
    pub shard_count: u64,
    pub start_ordinal: u64,
    pub end_ordinal_exclusive: u64,
    pub processed_actions: u64,
    pub actions_computed_this_run: u64,
    pub complete_declared_space: bool,
    pub block_size: u64,
    pub block_count: usize,
    pub threads_used: usize,
    pub elapsed_seconds: f64,
    pub throughput_actions_per_second: f64,
    pub checkpoint_directory: Option<String>,
    pub checkpoint_blocks_reused: usize,
    pub survivor_export_directory: Option<String>,
    pub survivor_record_format: Option<String>,
    pub gate_counts: BTreeMap<GateCode, u64>,
    pub survivor_count: u64,
    pub survivor_samples: Vec<CandidateSample>,
    pub rejection_witnesses: BTreeMap<GateCode, CandidateSample>,
    pub rejection_witness_rule: String,
    pub blocks_root_sha256: String,
    pub blocks: Vec<BlockManifest>,
    pub observational_data_opened: bool,
    pub deferred: Vec<String>,
}

fn basis_digest(basis: &[BasisTerm]) -> String {
    let payload = serde_json::to_vec(basis).expect("serialize basis");
    hex(&Sha256::digest(payload))
}

fn checkpoint_path(directory: &Path, block_index: u64, start: u64, end: u64) -> PathBuf {
    directory.join(format!("block-{block_index:08}-{start}-{end}.json"))
}

fn load_checkpoint(
    directory: &Path,
    block_index: u64,
    start: u64,
    end: u64,
) -> Option<BlockResult> {
    let path = checkpoint_path(directory, block_index, start, end);
    let bytes = fs::read(path).ok()?;
    let result: BlockResult = serde_json::from_slice(&bytes).ok()?;
    if result.manifest.block_index == block_index
        && result.manifest.start_ordinal == start
        && result.manifest.end_ordinal_exclusive == end
        && result.manifest.processed == end - start
    {
        Some(result)
    } else {
        None
    }
}

fn save_checkpoint(directory: &Path, result: &BlockResult) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let target = checkpoint_path(
        directory,
        result.manifest.block_index,
        result.manifest.start_ordinal,
        result.manifest.end_ordinal_exclusive,
    );
    let temporary = target.with_extension("json.tmp");
    let payload = serde_json::to_vec(result).map_err(|error| error.to_string())?;
    fs::write(&temporary, payload).map_err(|error| error.to_string())?;
    fs::rename(&temporary, &target).map_err(|error| error.to_string())?;
    Ok(())
}

fn save_survivor_block(directory: &Path, result: &mut BlockResult) -> Result<(), String> {
    fs::create_dir_all(directory).map_err(|error| error.to_string())?;
    let filename = format!(
        "survivors-{:08}-{}-{}.bin",
        result.manifest.block_index,
        result.manifest.start_ordinal,
        result.manifest.end_ordinal_exclusive
    );
    let target = directory.join(&filename);
    let temporary = target.with_extension("bin.tmp");
    let mut payload = Vec::with_capacity(48 + result.survivor_records.len() * 24);
    payload.extend_from_slice(b"SGSURV2\0");
    payload.extend_from_slice(&1_u16.to_le_bytes());
    payload.extend_from_slice(&24_u16.to_le_bytes());
    payload.extend_from_slice(&result.manifest.block_index.to_le_bytes());
    payload.extend_from_slice(&result.manifest.start_ordinal.to_le_bytes());
    payload.extend_from_slice(&result.manifest.end_ordinal_exclusive.to_le_bytes());
    payload.extend_from_slice(&(result.survivor_records.len() as u64).to_le_bytes());
    for survivor in &result.survivor_records {
        payload.extend_from_slice(&survivor.ordinal.to_le_bytes());
        payload.push(survivor.term_ids.len() as u8);
        payload.push(survivor.sign_mask);
        payload.extend_from_slice(&0_u16.to_le_bytes());
        for position in 0..6 {
            payload.extend_from_slice(
                &survivor
                    .term_ids
                    .get(position)
                    .copied()
                    .unwrap_or(u16::MAX)
                    .to_le_bytes(),
            );
        }
    }
    let file_sha256 = hex(&Sha256::digest(&payload));
    fs::write(&temporary, &payload).map_err(|error| error.to_string())?;
    if target.exists() {
        fs::remove_file(&target).map_err(|error| error.to_string())?;
    }
    fs::rename(&temporary, &target).map_err(|error| error.to_string())?;
    result.manifest.survivor_export = Some(SurvivorBlockManifest {
        file: filename,
        record_format: "SGSURV2/1 little-endian fixed-24-byte".into(),
        record_count: result.survivor_records.len() as u64,
        file_size_bytes: payload.len() as u64,
        file_sha256,
    });
    result.survivor_records.clear();
    Ok(())
}

pub fn run_generator(config: &GeneratorConfig, options: &RunOptions) -> RunManifest {
    config.validate().expect("valid generator config");
    assert!(options.start_ordinal <= options.end_ordinal_exclusive);
    assert!(options.block_size > 0);
    let total = total_search_count(config);
    assert!(options.end_ordinal_exclusive <= total);
    let basis = Arc::new(build_basis(config.basis_count));
    let hessians = Arc::new(precompute_hessians(config, &basis));
    let config = Arc::new(config.clone());

    let mut ranges = Vec::new();
    let mut cursor = options.start_ordinal;
    while cursor < options.end_ordinal_exclusive {
        let absolute_block = cursor / options.block_size;
        let next_boundary = (absolute_block + 1).saturating_mul(options.block_size);
        let end = next_boundary.min(options.end_ordinal_exclusive);
        ranges.push((absolute_block, cursor, end));
        cursor = end;
    }

    let mut reused_results = Vec::new();
    let mut pending_ranges = Vec::new();
    for (block_index, start, end) in ranges {
        let checkpoint = if options.survivor_directory.is_none() {
            options
                .checkpoint_directory
                .as_deref()
                .and_then(|directory| load_checkpoint(directory, block_index, start, end))
        } else {
            None
        };
        if let Some(result) = checkpoint {
            reused_results.push(result);
        } else {
            pending_ranges.push((block_index, start, end));
        }
    }
    let checkpoint_blocks_reused = reused_results.len();
    let reused_actions: u64 = reused_results
        .iter()
        .map(|result| result.manifest.processed)
        .sum();
    let work = Arc::new(pending_ranges);
    let next = Arc::new(AtomicUsize::new(0));
    let results: Arc<Mutex<Vec<BlockResult>>> = Arc::new(Mutex::new(reused_results));
    let threads = options.threads.max(1).min(work.len().max(1));
    let checkpoint_directory = Arc::new(options.checkpoint_directory.clone());
    let survivor_directory = Arc::new(options.survivor_directory.clone());
    let started = Instant::now();
    thread::scope(|scope| {
        for _ in 0..threads {
            let work = Arc::clone(&work);
            let next = Arc::clone(&next);
            let results = Arc::clone(&results);
            let config = Arc::clone(&config);
            let basis = Arc::clone(&basis);
            let hessians = Arc::clone(&hessians);
            let checkpoint_directory = Arc::clone(&checkpoint_directory);
            let survivor_directory = Arc::clone(&survivor_directory);
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= work.len() {
                        break;
                    }
                    let (block_index, start, end) = work[index];
                    let mut result = process_block(
                        block_index,
                        start,
                        end,
                        &config,
                        &basis,
                        &hessians,
                        survivor_directory.is_some(),
                    );
                    if let Some(directory) = survivor_directory.as_deref() {
                        save_survivor_block(directory, &mut result)
                            .expect("write compact survivor block");
                    }
                    if let Some(directory) = checkpoint_directory.as_deref() {
                        save_checkpoint(directory, &result).expect("write checkpoint block");
                    }
                    results.lock().unwrap().push(result);
                }
            });
        }
    });
    let elapsed = started.elapsed().as_secs_f64();
    let mut results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    results.sort_by_key(|result| result.manifest.start_ordinal);

    let mut gate_counts: BTreeMap<GateCode, u64> = BTreeMap::new();
    let mut samples = Vec::new();
    let mut rejection_witnesses: BTreeMap<GateCode, CandidateSample> = BTreeMap::new();
    let mut root = Sha256::new();
    root.update(b"SIGMA-GENERATOR-V2-ROOT\0");
    for result in &results {
        for (gate, count) in &result.gate_counts {
            *gate_counts.entry(*gate).or_insert(0) += count;
        }
        for sample in &result.samples {
            if sample.gate == GateCode::SurviveSampledStatic {
                samples.push(sample.clone());
            } else {
                let replace = rejection_witnesses
                    .get(&sample.gate)
                    .is_none_or(|existing| sample.candidate_id < existing.candidate_id);
                if replace {
                    rejection_witnesses.insert(sample.gate, sample.clone());
                }
            }
        }
        root.update(result.manifest.block_index.to_le_bytes());
        root.update(result.manifest.start_ordinal.to_le_bytes());
        root.update(result.manifest.end_ordinal_exclusive.to_le_bytes());
        root.update(result.manifest.digest_sha256.as_bytes());
    }
    samples.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    samples.truncate(config.sample_limit);
    let processed = options.end_ordinal_exclusive - options.start_ordinal;
    let computed = processed - reused_actions;
    let survivors = *gate_counts
        .get(&GateCode::SurviveSampledStatic)
        .unwrap_or(&0);
    let complete = options.start_ordinal == 0 && options.end_ordinal_exclusive == total;

    RunManifest {
        schema_version: "sigma-generator-v2-manifest-1.0".into(),
        generator_version: env!("CARGO_PKG_VERSION").into(),
        protocol_version: config.protocol_version.clone(),
        scope_claim: config.scope_claim.clone(),
        config_sha256: options.config_sha256.clone(),
        basis_count: config.basis_count,
        basis_library_sha256: basis_digest(&basis),
        max_action_terms: config.max_action_terms,
        coefficient_alphabet: config.coefficient_alphabet.clone(),
        coefficient_semantics: format!(
            "Every sparse term has coefficient +/- {}; signs are structural and do not add independent constants.",
            config.shared_coupling
        ),
        total_declared_actions: total,
        shard_index: options.shard_index,
        shard_count: options.shard_count,
        start_ordinal: options.start_ordinal,
        end_ordinal_exclusive: options.end_ordinal_exclusive,
        processed_actions: processed,
        actions_computed_this_run: computed,
        complete_declared_space: complete,
        block_size: options.block_size,
        block_count: results.len(),
        threads_used: threads,
        elapsed_seconds: elapsed,
        throughput_actions_per_second: if elapsed > 0.0 {
            computed as f64 / elapsed
        } else {
            0.0
        },
        checkpoint_directory: options
            .checkpoint_directory
            .as_ref()
            .map(|path| path.display().to_string()),
        checkpoint_blocks_reused,
        survivor_export_directory: options
            .survivor_directory
            .as_ref()
            .map(|path| path.display().to_string()),
        survivor_record_format: options
            .survivor_directory
            .as_ref()
            .map(|_| "SGSURV2/1 little-endian fixed-24-byte".into()),
        gate_counts,
        survivor_count: survivors,
        survivor_samples: samples,
        rejection_witnesses,
        rejection_witness_rule: "Each candidate receives exactly one first-failing structural or sampled-static gate; block hashes commit to candidate hash plus gate code.".into(),
        blocks_root_sha256: hex(&root.finalize()),
        blocks: results.into_iter().map(|result| result.manifest).collect(),
        observational_data_opened: false,
        deferred: vec![
            "global tensor static convexity beyond the frozen radial sample".into(),
            "automatic covariant tensor-action variation".into(),
            "ADM/Dirac constraint algebra and physical degree count".into(),
            "principal symbols and characteristic cones".into(),
            "GR/Solar and audited measurement gates".into(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(basis_count: usize, max_action_terms: usize) -> GeneratorConfig {
        GeneratorConfig {
            protocol_version: "TEST".into(),
            scope_claim: "test".into(),
            basis_count,
            max_action_terms,
            coefficient_alphabet: vec![-1, 1],
            shared_coupling: "epsilon".into(),
            coupling_magnitude: 0.1,
            convexity_tolerance: 1e-9,
            convexity_samples: ConvexitySamples {
                d: vec![0.1, 1.0, 10.0],
                p: vec![0.0, 0.5, 1.0],
                state: vec![0.0, 0.5, 1.0],
            },
            maximum_universal_constants: 5,
            universal_constants: vec!["a".into(), "L".into(), "Z0".into(), "epsilon".into()],
            sample_limit: 4,
            observational_data_opened: false,
        }
    }

    #[test]
    fn billion_space_count_is_exact() {
        assert_eq!(total_search_count(&config(50, 6)), 1_088_651_720);
    }

    #[test]
    fn ordinal_round_trip_boundaries_are_unique() {
        let config = config(6, 3);
        let total = total_search_count(&config);
        let mut encodings = std::collections::BTreeSet::new();
        for ordinal in 0..total {
            let decoded = decode_ordinal(&config, ordinal);
            assert!(encodings.insert((decoded.term_ids, decoded.sign_mask)));
        }
        assert_eq!(encodings.len() as u64, total);
    }

    #[test]
    fn basis_is_dimensionless_and_unique() {
        let basis = build_basis(50);
        let expressions: std::collections::BTreeSet<_> =
            basis.iter().map(|term| term.expression.clone()).collect();
        assert_eq!(expressions.len(), 50);
        assert!(
            basis
                .iter()
                .all(|term| term.dimension_l == 0 && term.dimension_t == 0)
        );
    }

    #[test]
    fn compiled_hessian_controls_match_positive_and_negative_elasticity() {
        let config = config(9, 1);
        let basis = build_basis(9);
        let hessians = precompute_hessians(&config, &basis);
        let negative_q = DecodedCandidate {
            ordinal: 0,
            term_ids: vec![3],
            sign_mask: 0,
        };
        let positive_q = DecodedCandidate {
            ordinal: 0,
            term_ids: vec![3],
            sign_mask: 1,
        };
        assert_eq!(
            evaluate_candidate(&negative_q, &basis, &config, &hessians),
            GateCode::RejectNegativeElasticity
        );
        assert_eq!(
            evaluate_candidate(&positive_q, &basis, &config, &hessians),
            GateCode::SurviveSampledStatic
        );
    }

    #[test]
    fn action_level_high_field_gate_keeps_exact_leading_cancellation() {
        let basis = build_basis(50);
        let cancelled = DecodedCandidate {
            ordinal: 0,
            term_ids: vec![3, 6, 25],
            sign_mask: 0b011,
        };
        let uncancelled = DecodedCandidate {
            ordinal: 0,
            term_ids: vec![3, 6, 25],
            sign_mask: 0b111,
        };
        assert_eq!(
            evaluate_structural(&cancelled, &basis),
            GateCode::SurviveSampledStatic
        );
        assert_eq!(
            evaluate_structural(&uncancelled, &basis),
            GateCode::RejectHighField
        );
    }
}
