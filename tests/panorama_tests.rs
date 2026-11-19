use image::{DynamicImage, GenericImageView, ImageReader, Rgba, RgbaImage};
use pure_sift::cli::{CliCommand, parse_cli_args};
use pure_sift::stitch::{
    BlendMode, PanoramaOptions, ProjectionMode, stitch_panorama, warp_image_cylindrical,
};

fn scale_exposure(img: &DynamicImage, factor: f32) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    let mut out = RgbaImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            out.put_pixel(
                x,
                y,
                Rgba([
                    ((p[0] as f32) * factor).round().clamp(0.0, 255.0) as u8,
                    ((p[1] as f32) * factor).round().clamp(0.0, 255.0) as u8,
                    ((p[2] as f32) * factor).round().clamp(0.0, 255.0) as u8,
                    255,
                ]),
            );
        }
    }
    DynamicImage::ImageRgba8(out)
}

#[test]
fn test_4_image_panorama_exposure_and_multiband_blending() {
    let base = ImageReader::open("docs/samples/aerial_west.jpg")
        .expect("open sample image")
        .decode()
        .expect("decode sample image");
    let (w, h) = base.dimensions();
    let tile_w = (w * 42) / 100;
    let step = (w - tile_w) / 3;

    let exposure_factors = [0.85f32, 1.0f32, 1.18f32, 0.92f32];
    let tiles: Vec<DynamicImage> = (0..4)
        .map(|i| {
            let crop = base.crop_imm(i as u32 * step, 0, tile_w, h);
            scale_exposure(&crop, exposure_factors[i])
        })
        .collect();

    let opts = PanoramaOptions {
        octaves: 3,
        scales: 3,
        threshold: 0.015,
        blend: BlendMode::MultiBand { levels: 4 },
        exposure_compensation: true,
        ..PanoramaOptions::default()
    };

    let result = stitch_panorama(&tiles, &opts).expect("4-tile panorama should stitch");
    assert_eq!(
        result.registered_indices.len(),
        4,
        "Expected all 4 overlapping tiles to register, got {:?}",
        result.registered_indices
    );
    assert!(
        result.image.width() > (w * 85) / 100,
        "Expected panorama width {} > 85% of source width {w}",
        result.image.width()
    );

    // Check solved exposure gains track inverse of applied exposure ratios relative to reference
    let ref_factor = exposure_factors[result.reference_index];
    for (i, &factor) in exposure_factors.iter().enumerate() {
        let expected_rel_gain = ref_factor / factor;
        let err = (result.gains[i] - expected_rel_gain).abs();
        assert!(
            err < 0.15,
            "Tile {i} gain {} deviated too far from expected {expected_rel_gain} (gains={:?})",
            result.gains[i],
            result.gains
        );
    }
}

#[test]
fn test_cylindrical_projection_and_cli_panorama_parsing() {
    let base = ImageReader::open("docs/samples/aerial_east.jpg")
        .expect("open east sample")
        .decode()
        .expect("decode east sample");
    let cyl = warp_image_cylindrical(&base, 650.0);
    assert_eq!(cyl.dimensions(), base.dimensions());
    // Center pixel must remain opaque after cylindrical warp
    assert_eq!(cyl.get_pixel(cyl.width() / 2, cyl.height() / 2)[3], 255);

    let cmd = parse_cli_args([
        "panorama",
        "a.jpg",
        "b.jpg",
        "c.jpg",
        "-o",
        "pano.png",
        "--projection",
        "cylindrical",
        "--focal=750",
        "--blend=feather",
        "--no-exposure",
    ])
    .expect("panorama args should parse");

    match cmd {
        CliCommand::Panorama(args) => {
            assert_eq!(args.image_paths.len(), 3);
            assert_eq!(args.output_path.to_str(), Some("pano.png"));
            assert_eq!(
                args.projection,
                ProjectionMode::Cylindrical { focal_px: 750.0 }
            );
            assert_eq!(args.blend, BlendMode::Feather);
            assert!(!args.exposure_compensation);
        }
        other => panic!("Unexpected command: {other:?}"),
    }
}
