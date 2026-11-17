#![no_main]

use libfuzzer_sys::fuzz_target;
use pure_sift::cli::parse_cli_args;

fuzz_target!(|data: &[u8]| {
    if let Ok(s) = std::str::from_utf8(data) {
        let args: Vec<&str> = s.split('\0').collect();
        let _ = parse_cli_args(args);
    }
});
