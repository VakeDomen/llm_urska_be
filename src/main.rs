use candle_core;
use tokenizers::Tokenizer;


const DEFAULT_PROMPT: &str = "My favorite theorem is ";


#[derive(Debug)]
enum Prompt {
    Interactive,
    Chat,
    One(String),
}



fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let _tokenizer = match tokenizer("models/llama3-8b/tokenizer.json") {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };
}

fn tokenizer(tokenizer: &str) -> anyhow::Result<Tokenizer> {
    let tokenizer_path = std::path::PathBuf::from(tokenizer);
    Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)
}