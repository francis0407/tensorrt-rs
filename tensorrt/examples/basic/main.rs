use tensorrt_rs::runtime::Logger;
use tensorrt_rs::builder::Builder;
use tensorrt_rs::engine::Engine;
use std::path::Path;
use tensorrt_rs::uff::{UffInputOrder, UffParser, UffFile};
use tensorrt_rs::dims::DimsCHW;

fn create_engine(uff_file: &UffFile) -> Engine {
    let logger = Logger::new();
    let builder = Builder::new(&logger);

    let uff_parser = UffParser::new();
    let dim = DimsCHW::new(1, 28, 28);
    uff_parser.register_input("in", dim, UffInputOrder::Nchw).unwrap();
    uff_parser.register_output("out").unwrap();
    uff_parser.parse(uff_file, builder.get_network()).unwrap();

    builder.build_cuda_engine()
}

fn main() {
    let uff_file = UffFile::new(Path::new("../assets/lenet5.uff")).unwrap();
    let engine = create_engine(&uff_file);

    println!("Engine number of bindings: {}", engine.get_nb_bindings());

    for binding_index in 0..engine.get_nb_bindings() {
        println!("Binding name at {}: {}", binding_index, engine.get_binding_name(binding_index).unwrap());
    }
}