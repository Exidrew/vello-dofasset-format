//! CPU-only mirror of `scene_builder.rs`, targeting `vello_cpu::RenderContext`
//! instead of `vello::Scene`. No `wgpu`/WebGPU involved anywhere in this
//! module — it exists so a `.dofasset` can be rasterised on devices where
//! WebGPU is unavailable or unreliable (see `webgpu-diagnostics.ts` on the
//! client side for the detection that picks this path).
//!
//! `vello` and `vello_cpu` share the same `kurbo`/`peniko` crate versions
//! (verified against the workspace lockfile), so `DofAsset`/`DrawCommand`
//! (format.rs) are consumed as-is — no type conversions, no duplicated
//! color/stroke-width resolution logic. The pure helpers (`resolve_color`,
//! `resolve_stroke_width`, `resolve_clip_mask`, `clip_mask_transform`,
//! `compute_net_offset`) are reused directly from `scene_builder` so the two
//! backends can never silently diverge on those semantics.
//!
//! Scope: body-part frames + base/delta z-order compositing (tiles, spell
//! icons, static UI panels). Accessory compositing (equipped
//! items/weapons/hats on player sprites) is intentionally NOT ported yet —
//! that's a separate follow-up increment once this core path is verified.

use std::collections::HashMap;

use vello::kurbo::{Affine, Stroke};
use vello::peniko::{Color, Fill, Gradient, ImageBrush};
use vello_cpu::{Image as CpuImage, ImageSource as CpuImageSource, Pixmap, RenderContext, Resources};

use crate::format::{DofAsset, DrawCommand, FillRule};
use crate::scene_builder::{
    clip_mask_transform, compute_net_offset, resolve_clip_mask, resolve_color,
    resolve_stroke_width,
};

/// A fully rasterised frame: straight (non-premultiplied) RGBA8 pixels plus
/// dimensions, ready to hand to `Texture.fromBuffer`-style APIs on the JS
/// side — no `ExternalSource`/GPU-texture plumbing required.
pub struct CpuFrame {
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8, straight alpha, 4 bytes/pixel.
    pub rgba: Vec<u8>,
}

/// Build one animation frame of `asset` entirely on the CPU.
///
/// Mirrors `scene_builder::build_frame_scene` for the no-accessory case:
/// same base/delta clip-offset math, same color-zone replacement, same
/// stroke-width resolution. `canvas_width`/`canvas_height` and
/// `bounds_offset` are expected to come from the same
/// `compute_animation_render_meta` call the GPU path already uses, so both
/// backends produce pixel-identical framing for a given asset+animation.
pub fn build_frame_pixmap_cpu(
    asset: &DofAsset,
    animation_name: &str,
    frame_index: usize,
    player_colors: Option<&[u32; 3]>,
    resolution: f32,
    bounds_offset: (f64, f64),
    canvas_width: u32,
    canvas_height: u32,
) -> Option<CpuFrame> {
    let anim_idx = *asset.animation_map.get(animation_name)?;
    let anim = &asset.animations[anim_idx];
    if anim.frame_ids.is_empty() {
        return None;
    }
    let actual_frame_idx = frame_index % anim.frame_ids.len();
    let global_frame_id = anim.frame_ids[actual_frame_idx] as usize;
    let frame = asset.frames.get(global_frame_id)?;

    let color_replacements =
        player_colors.map(|colors| crate::color::build_color_replacements(&asset.color_zones, colors));

    let scale = resolution as f64;

    let has_base = anim.base_frame_id != u32::MAX;
    let base_frame_opt = if has_base {
        asset.frames.get(anim.base_frame_id as usize)
    } else {
        None
    };

    let frame_clip_offset = Affine::translate((
        -(frame.clip_rect[0] as f64),
        -(frame.clip_rect[1] as f64),
    ));

    let (base_clip_offset, delta_clip_offset) = if let Some(base_frame) = base_frame_opt {
        let base_co = Affine::translate((
            -(base_frame.clip_rect[0] as f64),
            -(base_frame.clip_rect[1] as f64),
        ));
        let base_net = compute_net_offset(base_frame, &asset.transforms);
        let delta_net = compute_net_offset(frame, &asset.transforms);
        let adj = Affine::translate((base_net.0 - delta_net.0, base_net.1 - delta_net.1));
        (base_co, adj * frame_clip_offset)
    } else {
        (frame_clip_offset, frame_clip_offset)
    };

    let has_base_below = has_base && anim.base_z_order == 0;
    let has_base_above = has_base && anim.base_z_order == 1;

    let width = canvas_width.max(1).min(u16::MAX as u32) as u16;
    let height = canvas_height.max(1).min(u16::MAX as u32) as u16;
    let mut ctx = RenderContext::new(width, height);
    let mut resources = Resources::new();

    // Outer composition transform: bounds_offset then scale, applied as the
    // base transform every body-part draw gets multiplied into — equivalent
    // to the GPU path's `scene.append(&sub, Some(outer))`.
    let outer = Affine::translate(bounds_offset) * Affine::scale(scale);

    if has_base_below {
        if let Some(base_frame) = base_frame_opt {
            render_frame_parts_cpu(
                &mut ctx,
                &mut resources,
                asset,
                base_frame,
                outer * base_clip_offset,
                &color_replacements,
                resolution,
            );
        }
    }

    render_frame_parts_cpu(
        &mut ctx,
        &mut resources,
        asset,
        frame,
        outer * delta_clip_offset,
        &color_replacements,
        resolution,
    );

    if has_base_above {
        if let Some(base_frame) = base_frame_opt {
            render_frame_parts_cpu(
                &mut ctx,
                &mut resources,
                asset,
                base_frame,
                outer * base_clip_offset,
                &color_replacements,
                resolution,
            );
        }
    }

    let mut pixmap = Pixmap::new(width, height);
    ctx.render(&mut pixmap, &mut resources);

    let straight = pixmap.take_unpremultiplied();
    let mut rgba = Vec::with_capacity(straight.len() * 4);
    for px in straight {
        rgba.push(px.r);
        rgba.push(px.g);
        rgba.push(px.b);
        rgba.push(px.a);
    }

    Some(CpuFrame {
        width: width as u32,
        height: height as u32,
        rgba,
    })
}

/// Render just the body parts of a frame (no accessories) — CPU counterpart
/// of `scene_builder::render_frame_parts`.
fn render_frame_parts_cpu(
    ctx: &mut RenderContext,
    resources: &mut Resources,
    asset: &DofAsset,
    frame: &crate::format::Frame,
    clip_offset: Affine,
    color_replacements: &Option<HashMap<u32, Color>>,
    resolution: f32,
) {
    for part_inst in &frame.parts {
        let Some(body_part) = asset.body_parts.get(part_inst.body_part_id as usize) else {
            continue;
        };
        let part_transform = match asset.transforms.get(part_inst.transform_id as usize) {
            Some(&t) => clip_offset * t,
            None => clip_offset,
        };
        for &cmd_id in &body_part.draw_command_ids {
            if let Some(cmd) = asset.draw_commands.get(cmd_id as usize) {
                render_draw_command_cpu(
                    ctx,
                    resources,
                    asset,
                    cmd,
                    part_transform,
                    color_replacements,
                    resolution,
                );
            }
        }
    }
}

/// CPU counterpart of `scene_builder::render_draw_command`. Same transform
/// math and same `resolve_color`/`resolve_stroke_width` calls as the GPU
/// path — only the final "hand it to the renderer" calls differ, because
/// `vello_cpu::RenderContext` is a stateful API (`set_transform`/`set_paint`
/// then `fill_path`) instead of `Scene::fill(fill, transform, brush, ...)`.
fn render_draw_command_cpu(
    ctx: &mut RenderContext,
    resources: &mut Resources,
    asset: &DofAsset,
    cmd: &DrawCommand,
    part_transform: Affine,
    color_replacements: &Option<HashMap<u32, Color>>,
    resolution: f32,
) {
    let clip_id = cmd.clip_mask_id();
    let clip_pushed = if let Some(mask_path) = resolve_clip_mask(asset, clip_id) {
        let mask_xform = part_transform * clip_mask_transform(asset, clip_id);
        ctx.set_transform(mask_xform);
        ctx.push_clip_path(mask_path);
        true
    } else {
        false
    };

    match cmd {
        DrawCommand::Fill { path_id, fill_rule, color, zone_id, transform, .. } => {
            if let Some(path) = asset.paths.get(*path_id as usize) {
                let final_transform = part_transform * *transform;
                let resolved_color = resolve_color(*color, *zone_id, color_replacements);
                ctx.set_transform(final_transform);
                ctx.set_fill_rule(to_cpu_fill(*fill_rule));
                ctx.set_paint(resolved_color);
                ctx.fill_path(path);
            }
        }
        DrawCommand::Stroke {
            path_id, color, zone_id, width, width_mode, line_cap, line_join, transform, ..
        } => {
            if let Some(path) = asset.paths.get(*path_id as usize) {
                let final_transform = part_transform * *transform;
                let resolved_color = resolve_color(*color, *zone_id, color_replacements);
                let stroke_width = resolve_stroke_width(*width, *width_mode, final_transform, resolution);
                let stroke = Stroke::new(stroke_width)
                    .with_caps(*line_cap)
                    .with_join(*line_join);
                ctx.set_transform(final_transform);
                ctx.set_stroke(stroke);
                ctx.set_paint(resolved_color);
                ctx.stroke_path(path);
            }
        }
        DrawCommand::PatternFill { path_id, image_id, pattern_transform, transform, fill_rule, .. } => {
            if let Some(path) = asset.paths.get(*path_id as usize) {
                if let Some(image) = asset.images.get(*image_id as usize) {
                    let final_transform = part_transform * *transform;
                    let brush = to_cpu_image_brush(image);
                    // `pattern_transform` positions the tiled pattern in its
                    // own space — it must NOT be folded into the path's
                    // transform (that would also warp the shape's geometry).
                    // `set_paint_transform` is vello_cpu's equivalent of the
                    // GPU path's separate `brush_transform` argument to
                    // `Scene::fill`.
                    ctx.set_transform(final_transform);
                    ctx.set_paint_transform(*pattern_transform);
                    ctx.set_fill_rule(to_cpu_fill(*fill_rule));
                    ctx.set_paint(brush);
                    ctx.fill_path(path);
                    ctx.reset_paint_transform();
                }
            }
        }
        DrawCommand::GradientFill {
            path_id, fill_rule, gradient_type, cx, cy, fx, fy, r, gradient_transform, stops, transform, ..
        } => {
            if let Some(path) = asset.paths.get(*path_id as usize) {
                let final_transform = part_transform * *transform;
                let grad_stops = stops.clone();
                let gradient = if *gradient_type == 0 {
                    Gradient::new_two_point_radial(
                        vello::kurbo::Point::new(*cx as f64, *cy as f64), 0_f32,
                        vello::kurbo::Point::new(*fx as f64, *fy as f64), *r,
                    ).with_stops(grad_stops.as_slice())
                } else {
                    Gradient::new_linear((*cx as f64, *cy as f64), (*fx as f64, *fy as f64))
                        .with_stops(grad_stops.as_slice())
                };
                // Same separate-transform-channel reasoning as PatternFill above.
                ctx.set_transform(final_transform);
                ctx.set_paint_transform(*gradient_transform);
                ctx.set_fill_rule(to_cpu_fill(*fill_rule));
                ctx.set_paint(gradient);
                ctx.fill_path(path);
                ctx.reset_paint_transform();
            }
        }
    }

    if clip_pushed {
        ctx.pop_clip_path();
    }
    // `resources` isn't actually touched by the inline `ImageSource::Pixmap`
    // path below — kept as a parameter so a future switch to the shared
    // image registry (for large, reused atlases) doesn't change call sites.
    let _ = resources;
}

fn to_cpu_fill(rule: FillRule) -> Fill {
    match rule {
        FillRule::NonZero => Fill::NonZero,
        FillRule::EvenOdd => Fill::EvenOdd,
    }
}

/// Convert a GPU-side `peniko::ImageBrush<ImageData>` (our `.dofasset`
/// pattern images, decoded once at load time by `format.rs`) into the CPU
/// renderer's `ImageBrush<ImageSource>`. Pixels are embedded inline
/// (`ImageSource::Pixmap`) rather than registered in the shared registry —
/// pattern fills in `.dofasset` sprites are small (leather-texture tiles),
/// so the extra `Arc` clone per draw is not worth the registry bookkeeping.
/// Mirrors `pattern::create_pattern_brush` (GPU path): pattern fills always
/// tile, regardless of whatever extend mode happens to be stored on the
/// decoded `ImageBrush` — forcing `Repeat` here is what fixed a ~60%
/// pixel-diff regression against the GPU reference render (ground tiles'
/// dirt/leather textures were clamping to the first tile instead of
/// repeating across the fill).
fn to_cpu_image_brush(image: &ImageBrush) -> CpuImage {
    let source = CpuImageSource::from_peniko_image_data(&image.image);
    CpuImage {
        image: source,
        sampler: image
            .sampler
            .with_x_extend(vello::peniko::Extend::Repeat)
            .with_y_extend(vello::peniko::Extend::Repeat),
    }
}
