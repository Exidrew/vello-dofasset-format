use image::{GenericImageView, Rgba, RgbaImage};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a = image::open(&args[1]).unwrap();
    let b = image::open(&args[2]).unwrap();
    let (w, h) = a.dimensions();
    assert_eq!((w, h), b.dimensions());
    let mut out = RgbaImage::new(w, h);
    let mut diffcount = 0u64;
    let mut maxdiff = 0i32;
    let mut sumdiff: u64 = 0;
    for y in 0..h {
        for x in 0..w {
            let pa = a.get_pixel(x, y);
            let pb = b.get_pixel(x, y);
            let mut d = 0i32;
            for i in 0..4 {
                let diff = (pa[i] as i32 - pb[i] as i32).abs();
                d = d.max(diff);
                sumdiff += diff as u64;
            }
            if d > 0 { diffcount += 1; }
            maxdiff = maxdiff.max(d);
            let amp = (d.min(255)) as u8;
            out.put_pixel(x, y, Rgba([amp, 0, 255 - amp, 255]));
        }
    }
    let total = (w * h) as u64;
    println!("pixels differing: {diffcount}/{total} ({:.3}%)", 100.0 * diffcount as f64 / total as f64);
    println!("max channel diff: {maxdiff}");
    println!("avg channel diff: {:.4}", sumdiff as f64 / (total * 4) as f64);
    out.save(&args[3]).unwrap();
    println!("wrote diff heatmap to {}", args[3]);
}
