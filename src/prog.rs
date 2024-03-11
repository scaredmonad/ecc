use crate::compile;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

// Because nobody needs structopt or clap. It's not ergonomic, we'd expect
// a tuple of HashMap (fields) and Vec (flags).
fn argparse() -> HashMap<String, String> {
    let args: Vec<String> = env::args().collect();
    let mut args_map = HashMap::new();

    let mut iter = args.iter().peekable();

    while let Some(arg) = iter.next() {
        if arg.starts_with('-') {
            let key = arg.trim_start_matches('-').to_string();
            if let Some(next_arg) = iter.peek() {
                // Check is a value or another flag.
                if next_arg.starts_with('-') {
                    // Next arg is a flag, current flag is boolean.
                    args_map.insert(key, "true".to_string());
                } else {
                    // Value arg.
                    let value = iter.next().unwrap().to_string();
                    args_map.insert(key, value);
                }
            } else {
                // No more args, current flag is boolean.
                args_map.insert(key, "true".to_string());
            }
        }
    }

    args_map
}

fn replace_extension<P: AsRef<Path>>(path: P, new_extension: &str) -> PathBuf {
    if let Some(parent) = path.as_ref().parent() {
        if let Some(file_stem) = path.as_ref().file_stem() {
            let mut new_path_buf = PathBuf::from(parent);
            new_path_buf.push(file_stem);
            new_path_buf.set_extension(new_extension);
            return new_path_buf;
        }
    }

    PathBuf::new()
}

fn write(input_path: &str, content: &str) {
    let file_name = replace_extension(input_path, "wat");
    let file_name = file_name
        .to_str()
        .expect("Couldn't cast output path to str.");

    match fs::write(file_name, content) {
        Ok(()) => println!("Wrote output to {}", file_name),
        Err(e) => eprintln!("Error: {}", e),
    }
}

pub(crate) fn compile_from_env() {
    let args = argparse();

    for (k, v) in &args {
        if v == "true" {
            // We know it's a flag.
            core::panic!("Flags not yet supported.")
        }

        if k == "i" {
            let src = fs::read_to_string(v).expect("Unable to read input file");
            write(v, &compile(&src).unwrap());
        }
    }
}
