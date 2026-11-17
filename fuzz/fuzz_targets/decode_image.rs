#![no_main]

use libfuzzer_sys::fuzz_target;
use pure_sift::cli::decode_validated_dynamic_image;

fuzz_target!(|data: &[u8]| {
    let _ = decode_validated_dynamic_image(data, true);
});
