use noface::config::get_config;

fn main() {
    let config = get_config();
    match config {
        Ok(c) => println!("{:?}", c),
        Err(err) => println!("{:?}", err),
    }
}
