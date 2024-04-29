use candle_core::Tensor;
use llm::{loader::load_bert_model, prompt::{prompt_model, Prompt}};
use logging::logger::init_logging;
use tokenizers::PaddingParams;
use wss::server::start_server;
use anyhow::{Error, Result};

mod llm;
mod config;
mod wss;
mod storage;
mod logging;

#[tokio::main]
async fn main() -> Result<()> {
    init_logging()?;
    start_server().await;
    Ok(())
}


