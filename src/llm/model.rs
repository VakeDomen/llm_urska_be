use candle_core::{quantized::gguf_file::Content, Device};
use candle_transformers::{generation::{LogitsProcessor, Sampling}, models::quantized_llama::ModelWeights};
use anyhow::Result;

use crate::config::{SEED, TEMPERATURE, TOP_K, TOP_P};

pub fn load_model(model_path: &str, device: &Device) -> Result<ModelWeights> {
    let model_path = std::path::PathBuf::from(model_path);
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    
    let model = Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    
    println!(
        "Loaded {:?} tensors in {:.2}s",
        model.tensor_infos.len(),
        start.elapsed().as_secs_f32(),
    );
    
    let weights = ModelWeights::from_gguf(model, &mut file, device)?;
    Ok(weights)
}


pub fn setup_logit_procesing() -> LogitsProcessor {
    let sampling = setup_sampling();
    LogitsProcessor::from_sampling(SEED, sampling)
}

fn setup_sampling() -> Sampling {
    if TEMPERATURE <= 0. {
        Sampling::ArgMax
    } else {
        match (TOP_K, TOP_P) {
            (None, None) => Sampling::All { temperature: TEMPERATURE },
            (Some(k), None) => Sampling::TopK { k, temperature: TEMPERATURE },
            (None, Some(p)) => Sampling::TopP { p, temperature: TEMPERATURE },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p,temperature:  TEMPERATURE },
        }
    }
}