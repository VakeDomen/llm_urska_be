use anyhow::{Error, Result};
use candle_core::Tensor;
use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;

use crate::{llm::loader::load_bert_model, logging::flush::{flush_message, FlushType}};



pub async fn generate_prompt_embedding(
    prompt: &str,
    mut websocket: Option<&mut WebSocketStream<TcpStream>>
) -> Result<Tensor> {
    flush_message("Loading embedding model...", &mut websocket, FlushType::Status).await?;
    let (model, tokenizer, device) = load_bert_model(Some(0))?;
    
    flush_message("Encoding embedding prompt...", &mut websocket, FlushType::Status).await?;
    let tokens = tokenizer
            .encode(prompt, true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();
    let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let token_type_ids = token_ids.zeros_like()?;
    
    flush_message("Generating word embeddings...", &mut websocket, FlushType::Status).await?;
    let embeddings = model.forward(&token_ids, &token_type_ids)?;


    // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
    flush_message("Pooling sentance embeddings...", &mut websocket, FlushType::Status).await?;
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
    let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
    
    let embeddings = normalize_l2(&embeddings)?;
    
    flush_message("Prompt embeddings generated!", &mut websocket, FlushType::Status).await?;
    Ok(embeddings)
} 

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
