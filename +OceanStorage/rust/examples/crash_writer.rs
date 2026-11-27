use ocean_storage::Engine;
fn main() {
    let path = std::env::args_os().nth(1).expect("database directory");
    let mut e = Engine::open(path).expect("open");
    e.put(b"crash-key", b"durable-value").expect("put");
}
