//! Smoke test for the CPU rasterizer path (`scene_builder_cpu`). Loads a
//! real `.dofasset` from disk and renders one frame through `vello_cpu` —
//! no `wgpu`/WebGPU anywhere in this binary — writing a PNG so the result
//! can be eyeballed against the GPU renderer's output (`cargo run --bin
//! dofasset-renderer`) for visual parity.
//!
//! Usage: cargo run --bin cpu-bench -- <input.dofasset> [--animation staticF]
//!        [--frame 0] [--colors 0xff0000,0x00ff00,0x0000ff] [--resolution 2]
//!        [--output cpu-render.png]

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use dofasset_renderer::{format, scene_builder, scene_builder_cpu};

struct Args {
    input: PathBuf,
    animation: String,
    frame: usize,
    colors: Option<[u32; 3]>,
    resolution: f32,
    output: PathBuf,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: cargo run --bin cpu-bench -- <input.dofasset> [--animation staticF] \
             [--frame 0] [--colors 0xff0000,0x00ff00,0x0000ff] [--resolution 2] [--output cpu-render.png]"
        );
        std::process::exit(1);
    }

    let mut result = Args {
        input: PathBuf::from(&args[1]),
        animation: "staticF".to_string(),
        frame: 0,
        colors: None,
        resolution: 2.0,
        output: PathBuf::from("cpu-render.png"),
    };

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--animation" => {
                result.animation = args.get(i + 1).cloned().unwrap_or_default();
                i += 2;
            }
            "--frame" => {
                result.frame = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--colors" => {
                if let Some(s) = args.get(i + 1) {
                    let parts: Vec<u32> = s
                        .split(',')
                        .filter_map(|p| {
                            let p = p.trim().trim_start_matches("0x").trim_start_matches("0X");
                            u32::from_str_radix(p, 16).ok()
                        })
                        .collect();
                    if parts.len() >= 3 {
                        result.colors = Some([parts[0], parts[1], parts[2]]);
                    }
                }
                i += 2;
            }
            "--resolution" => {
                result.resolution = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(2.0);
                i += 2;
            }
            "--output" => {
                result.output = PathBuf::from(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    result
}

fn main() {
    let args = parse_args();

    if !args.input.exists() {
        eprintln!("Input file not found: {}", args.input.display());
        std::process::exit(1);
    }

    println!("Loading: {}", args.input.display());
    let load_start = Instant::now();
    let file_data = fs::read(&args.input).expect("Failed to read file");
    let asset = format::load(&file_data);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "Loaded in {load_ms:.2}ms — {} animations, {} body parts, {} frames",
        asset.animations.len(),
        asset.body_parts.len(),
        asset.frames.len()
    );

    if asset.animation_map.get(&args.animation).is_none() {
        eprintln!("Animation '{}' not found. Available:", args.animation);
        for anim in &asset.animations {
            eprintln!("  {} ({} frames)", anim.name, anim.frame_ids.len());
        }
        std::process::exit(1);
    }

    let meta = scene_builder::compute_animation_render_meta(
        &asset,
        &args.animation,
        args.resolution,
        &[],
    );
    println!(
        "Canvas: {}x{} (CPU path, no wgpu/WebGPU)",
        meta.canvas_width, meta.canvas_height
    );

    let render_start = Instant::now();
    let frame = scene_builder_cpu::build_frame_pixmap_cpu(
        &asset,
        &args.animation,
        args.frame,
        args.colors.as_ref(),
        args.resolution,
        (meta.bounds_offset_x, meta.bounds_offset_y),
        meta.canvas_width,
        meta.canvas_height,
    )
    .expect("CPU render failed");
    let render_ms = render_start.elapsed().as_secs_f64() * 1000.0;
    println!("Rendered in {render_ms:.2}ms (CPU, single frame)");

    image::save_buffer(
        &args.output,
        &frame.rgba,
        frame.width,
        frame.height,
        image::ColorType::Rgba8,
    )
    .expect("Failed to write PNG");
    println!("Wrote {}", args.output.display());
}
