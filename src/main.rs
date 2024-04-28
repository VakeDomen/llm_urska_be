
use crate::{
    config::DEFAULT_PROMPT, 
    llm::prompt::{prompt_model, Prompt}
};

mod llm;
mod config;


fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    loop {
        let start = std::time::Instant::now();
        let prompt = Prompt::One(DEFAULT_PROMPT.to_string());

        match prompt_model(prompt) {
            Ok(out) => println!("[{:?}]", 
            start.elapsed().as_secs_f32()),
            Err(e) => panic!("Can't prompt model: {:#?}", e),
        }
    }
}



