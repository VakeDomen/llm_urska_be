use candle_transformers::generation::{LogitsProcessor, Sampling};
use crate::config::{SEED, TEMPERATURE, TOP_K, TOP_P};


#[derive(Debug)]
pub enum ModelArchitecture {
    Llama3,
    Mixtral,
}


/// Sets up a logits processor based on predefined settings and sampling strategy.
///
/// # Returns
/// An instance of `LogitsProcessor` configured with a specific sampling strategy.
pub fn setup_logit_procesing() -> LogitsProcessor {
    let sampling = setup_sampling();
    LogitsProcessor::from_sampling(SEED, sampling)
}

/// Configures the sampling strategy based on predefined temperature and probability settings.
///
/// # Returns
/// A `Sampling` variant configured according to the global temperature, TOP_K, and TOP_P settings.
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