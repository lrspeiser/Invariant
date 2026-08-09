use serde_json::to_writer_pretty;
use sha2::{Digest, Sha256};
use sigma_generator_v2::{
    GeneratorConfig, RunOptions, build_basis, run_generator, total_search_count,
};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::PathBuf;
use std::thread;

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn usage() -> ! {
    eprintln!(
        "Usage:\n  sigma-generator-v2 count --config PATH\n  sigma-generator-v2 basis --config PATH --output PATH\n  sigma-generator-v2 run --config PATH --output PATH [--limit N] [--start N] [--threads N] [--block-size N] [--checkpoint-dir PATH] [--survivor-dir PATH] [--shard-count N --shard-index N]"
    );
    std::process::exit(2);
}

fn parse_args() -> (String, BTreeMap<String, String>) {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| usage());
    let mut values = BTreeMap::new();
    while let Some(flag) = args.next() {
        if !flag.starts_with("--") {
            usage();
        }
        let value = args.next().unwrap_or_else(|| usage());
        values.insert(flag, value);
    }
    (command, values)
}

fn required(values: &BTreeMap<String, String>, name: &str) -> String {
    values.get(name).cloned().unwrap_or_else(|| usage())
}

fn number(values: &BTreeMap<String, String>, name: &str, default: u64) -> u64 {
    values
        .get(name)
        .map(|value| value.parse().unwrap_or_else(|_| usage()))
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (command, values) = parse_args();
    let config_path = PathBuf::from(required(&values, "--config"));
    let config_bytes = fs::read(&config_path)?;
    let config: GeneratorConfig = serde_json::from_slice(&config_bytes)?;
    config.validate().map_err(io::Error::other)?;
    let config_sha256 = hex(&Sha256::digest(&config_bytes));
    let total = total_search_count(&config);

    match command.as_str() {
        "count" => {
            println!("protocol={}", config.protocol_version);
            println!("basis_count={}", config.basis_count);
            println!("max_action_terms={}", config.max_action_terms);
            println!("total_declared_actions={total}");
            println!("config_sha256={config_sha256}");
        }
        "basis" => {
            let output = PathBuf::from(required(&values, "--output"));
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(&output)?;
            to_writer_pretty(&mut file, &build_basis(config.basis_count))?;
            writeln!(&mut file)?;
            println!("basis={}", output.display());
        }
        "run" => {
            let output = PathBuf::from(required(&values, "--output"));
            let shard_count = number(&values, "--shard-count", 1);
            let shard_index = number(&values, "--shard-index", 0);
            if shard_count == 0 || shard_index >= shard_count {
                return Err(io::Error::other("invalid shard index/count").into());
            }
            let shard_start = (total as u128 * shard_index as u128 / shard_count as u128) as u64;
            let shard_end =
                (total as u128 * (shard_index + 1) as u128 / shard_count as u128) as u64;
            let start = number(&values, "--start", shard_start).max(shard_start);
            let limit = number(&values, "--limit", shard_end - start);
            let end = start.saturating_add(limit).min(shard_end);
            let threads = number(
                &values,
                "--threads",
                thread::available_parallelism()
                    .map(|n| n.get() as u64)
                    .unwrap_or(1),
            ) as usize;
            let options = RunOptions {
                start_ordinal: start,
                end_ordinal_exclusive: end,
                block_size: number(&values, "--block-size", 65_536),
                threads,
                shard_index,
                shard_count,
                config_sha256,
                checkpoint_directory: values.get("--checkpoint-dir").map(PathBuf::from),
                survivor_directory: values.get("--survivor-dir").map(PathBuf::from),
            };
            let manifest = run_generator(&config, &options);
            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(&output)?;
            to_writer_pretty(&mut file, &manifest)?;
            writeln!(&mut file)?;
            println!("processed_actions={}", manifest.processed_actions);
            println!(
                "actions_computed_this_run={}",
                manifest.actions_computed_this_run
            );
            println!(
                "checkpoint_blocks_reused={}",
                manifest.checkpoint_blocks_reused
            );
            println!("survivor_count={}", manifest.survivor_count);
            println!(
                "throughput_actions_per_second={:.0}",
                manifest.throughput_actions_per_second
            );
            println!("blocks_root_sha256={}", manifest.blocks_root_sha256);
            println!("manifest={}", output.display());
        }
        _ => usage(),
    }
    Ok(())
}
