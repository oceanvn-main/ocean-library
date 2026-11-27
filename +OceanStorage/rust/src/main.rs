use ocean_storage::Engine;
use std::io::{self, BufRead, Write};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
fn decode(text: &str) -> Result<Vec<u8>, String> {
    if !text.len().is_multiple_of(2) {
        return Err("odd hex length".into());
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let a = (pair[0] as char).to_digit(16).ok_or("invalid hex")?;
            let b = (pair[1] as char).to_digit(16).ok_or("invalid hex")?;
            Ok(((a << 4) | b) as u8)
        })
        .collect()
}
fn request(engine: &mut Engine, line: &str) -> Result<(String, bool), String> {
    let p: Vec<_> = line.split('\t').collect();
    match p.as_slice() {
        ["PUT", key, value] => {
            engine
                .put(&decode(key)?, &decode(value)?)
                .map_err(|e| e.to_string())?;
            Ok(("OK\n".into(), false))
        }
        ["GET", key] => Ok((
            match engine.get(&decode(key)?).map_err(|e| e.to_string())? {
                Some(v) => format!("VALUE\t{}\n", hex(&v)),
                None => "MISSING\n".into(),
            },
            false,
        )),
        ["DELETE", key] => {
            engine.delete(&decode(key)?).map_err(|e| e.to_string())?;
            Ok(("OK\n".into(), false))
        }
        ["SCAN", start, end] => {
            let end = if *end == "*" {
                None
            } else {
                Some(decode(end)?)
            };
            let rows = engine
                .scan(&decode(start)?, end.as_deref())
                .map_err(|e| e.to_string())?;
            let mut output = String::new();
            for (key, value) in rows {
                output.push_str(&format!("ROW\t{}\t{}\n", hex(&key), hex(&value)));
            }
            output.push_str("END\n");
            Ok((output, false))
        }
        ["INSPECT"] => Ok((
            format!(
                "TEXT\t{}\n",
                hex(engine.inspect().map_err(|e| e.to_string())?.as_bytes())
            ),
            false,
        )),
        ["CLOSE"] => Ok(("OK\n".into(), true)),
        _ => Err("invalid request".into()),
    }
}
fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 || args[1] != "serve" {
        return Err("usage: ocean-storage serve <database-directory>".into());
    }
    let mut engine = Engine::open(&args[2])?;
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    output.write_all(b"READY\t1\n")?;
    output.flush()?;
    loop {
        // Read at most 4097 bytes; discard oversized requests without allocating their full size.
        let mut line = Vec::new();
        let mut oversized = false;
        loop {
            let available = input.fill_buf()?;
            if available.is_empty() {
                break;
            }
            let count = available
                .iter()
                .position(|b| *b == b'\n')
                .map_or(available.len(), |i| i + 1);
            if line.len() + count <= 4096 && !oversized {
                line.extend_from_slice(&available[..count]);
            } else {
                oversized = true;
            }
            let done = available[count - 1] == b'\n';
            input.consume(count);
            if done {
                break;
            }
        }
        if line.is_empty() && !oversized {
            break;
        }
        let result = if oversized {
            Err("request exceeds 4096 bytes".into())
        } else {
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }
            std::str::from_utf8(&line)
                .map_err(|_| "request must be ASCII".to_string())
                .and_then(|s| request(&mut engine, s))
        };
        let (reply, close) =
            result.unwrap_or_else(|e| (format!("ERR\t{}\n", hex(e.as_bytes())), false));
        output.write_all(reply.as_bytes())?;
        output.flush()?;
        if close {
            break;
        }
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        if error
            .downcast_ref::<io::Error>()
            .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
        {
            return;
        }
        eprintln!("{error}");
        std::process::exit(1);
    }
}
