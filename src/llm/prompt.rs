use std::io::Write;
use anyhow::{Error, Result};
use candle_core::Tensor;
use log::{debug, info};
use tokio::net::TcpStream;
use tokio_tungstenite::WebSocketStream;

use crate::{
    config::{
        HYDE_MODEL_ARCITECTURE, HYDE_SYSTEM_MSG, MODEL_ARCITECTURE, REPEAT_LAST_N, REPEAT_PENALTY, SAMPLE_LEN, SPLIT_PROPMT, SYSTEM_MSG, SYSTEM_RAG_MSG, VERBOSE_PROMPT
    }, llm::{
        loader::{
            assign_model, ModelSelector, MODEL1, MODEL2
        },
        model::setup_logit_procesing,
        tokenizer::TokenOutputStream
    }, logging::flush::{flush_message, FlushType}
};

use super::loader::{HYDE_MODEL1, HYDE_MODEL2};

/// Represents different types of prompts that can be processed by the system.
///
/// This enum categorizes prompts into plain questions and questions with associated passages for Retrieval-Augmented Generation (RAG).
#[derive(Debug)]
pub enum Prompt {
    /// Represents a plain question without associated contextual information.
    PlainQuestion(String),
    /// Represents a question accompanied by a set of relevant passages, used for RAG.
    RagPrompt(String, Vec<(String, String)>),
    /// Represents a question that should generate imaginary context information.
    Hyde(String)
}

/// Parses a `Prompt` into a formatted string that can be used as input to an LLM.
///
/// Depending on the type of `Prompt`, this function formats it differently to align with the requirements of the LLM processing.
///
/// # Parameters
/// - `prompt`: The `Prompt` enum containing either a plain question or a RAG type question with passages.
///
/// # Returns
/// Returns a `Result` with a formatted string suitable for LLM input if successful, or an error if the formatting fails.
pub fn parse_prompt_to_raw(prompt: Prompt) -> String {
    match prompt {
        Prompt::PlainQuestion(prompt) => plain_question_prompt(prompt),
        Prompt::RagPrompt(prompt, passages) => rag_prompt(prompt, passages),
        Prompt::Hyde(prompt) => hyde_prompt(prompt),
    }
}

pub fn plain_question_prompt(prompt: String) -> String {
    match MODEL_ARCITECTURE {
        super::model::ModelArchitecture::Llama3 => llama3_prompt(SYSTEM_MSG, prompt),
        super::model::ModelArchitecture::Mixtral => mixtral_prompt(SYSTEM_MSG, prompt),
    }
}

pub fn hyde_prompt(prompt: String) -> String {
    match HYDE_MODEL_ARCITECTURE {
        super::model::ModelArchitecture::Llama3 => llama3_prompt(HYDE_SYSTEM_MSG, prompt),
        super::model::ModelArchitecture::Mixtral => mixtral_prompt(HYDE_SYSTEM_MSG, prompt),
    }
}

pub fn rag_prompt(prompt: String, passages: Vec<(String, String)>) -> String {
    match MODEL_ARCITECTURE {
        super::model::ModelArchitecture::Llama3 => llama3_rag_prompt(prompt, passages),
        super::model::ModelArchitecture::Mixtral => mixtral_rag_prompt(prompt, passages),
    }
}

/// Formats a plain user message for processing by a Llama3 model.
///
/// This function prepares a plain text message by incorporating predefined system messages for consistency in LLM input.
///
/// # Parameters
/// - `user_msg`: The user's message or question to be formatted.
///
/// # Returns
/// Returns a string formatted specifically for input to a language model, adhering to a specified interaction schema.
pub fn llama3_prompt(system_msg: &str, user_msg: String) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
        system_msg,
        user_msg
    )
}

/// Formats a plain user message for processing by a Mixtral8x7b model.
///
/// This function prepares a plain text message by incorporating predefined system messages for consistency in LLM input.
///
/// # Parameters
/// - `user_msg`: The user's message or question to be formatted.
///
/// # Returns
/// Returns a string formatted specifically for input to a language model, adhering to a specified interaction schema.
pub fn mixtral_prompt(system_msg: &str, user_msg: String) -> String {
    format!(
        "<s>[INST] {}\n\n{} [/INST]", 
        system_msg,
        user_msg
    )
}

/// Formats a RAG prompt by including relevant passages and a user message for processing by a Llama3 model.
///
/// This function constructs a prompt that incorporates passages for context along with the user's question, providing enriched input for the LLM.
///
/// # Parameters
/// - `user_msg`: The user's question to be included.
/// - `passages`: A list of relevant passages providing context to the question.
///
/// # Returns
/// Returns a string formatted to include multiple context passages and the user's question, specifically designed for input to a language model.
pub fn llama3_rag_prompt(user_msg: String, passages: Vec<(String, String)>) -> String {
    let passage_string = passages
        .iter()
        .map(|p| p.1.clone())
        .collect::<Vec<String>>()
        .join("\n# Passage: ");

    let user_msg = format!(
        "# Passage:\n{}\n# Student question: \n{}",
        passage_string,
        user_msg
    );
    llama3_prompt(SYSTEM_RAG_MSG, user_msg)
}


/// Formats a RAG prompt by including relevant passages and a user message for processing by a Mixtral model.
///
/// This function constructs a prompt that incorporates passages for context along with the user's question, providing enriched input for the LLM.
///
/// # Parameters
/// - `user_msg`: The user's question to be included.
/// - `passages`: A list of relevant passages providing context to the question.
///
/// # Returns
/// Returns a string formatted to include multiple context passages and the user's question, specifically designed for input to a language model.
pub fn mixtral_rag_prompt(user_msg: String, passages: Vec<(String, String)>) -> String {
    let passage_string = passages
        .iter()
        .map(|p| p.1.clone())
        .collect::<Vec<String>>()
        .join("\n# Passage: ");

    let user_msg = format!(
        "# Passage:\n{}\n# Student question: \n{}",
        passage_string,
        user_msg
    );
    mixtral_prompt(SYSTEM_RAG_MSG, user_msg)
}


pub fn get_eos_token(hyde: bool) -> String {
    let architecutre = match hyde {
        true => HYDE_MODEL_ARCITECTURE,
        false => MODEL_ARCITECTURE,
    };
    match architecutre {
        super::model::ModelArchitecture::Llama3 => "<|eot_id|>".to_owned(),
        super::model::ModelArchitecture::Mixtral => "</s>".to_owned(),
    }
}


/// Generates a response from a language model based on the input prompt, and optionally handles real-time updates via WebSocket.
///
/// This function first assigns a model based on a toggle, then tokenizes and processes the input prompt for the model.
/// It handles the complexities of input preparation, model invocation, and output generation, ensuring the response
/// is appropriately clipped and generated according to specified parameters.
///
/// # Parameters
/// - `prompt`: The input prompt provided in one of the supported formats, either plain or with context.
/// - `websocket`: An optional mutable reference to a WebSocket stream for real-time interaction updates.
///
/// # Returns
/// Returns a `Result` with the generated response as a string if successful. Any issues during processing will
/// result in errors being returned.
///
/// # Errors
/// - Returns an error if tokenization, model execution, or response generation fails.
/// - Any interaction with the WebSocket that fails will also propagate as an error.
pub async fn prompt_model(
    prompt: Prompt,
    hyde: bool,
    mut websocket: Option<&mut WebSocketStream<TcpStream>>
) -> Result<String> {

    flush_message("Assigning LLM model...", &mut websocket, FlushType::Status).await?;

    let model_selector = assign_model(hyde).await;
    let mut loaded_model = match model_selector {
        ModelSelector::First => MODEL1.lock().await,
        ModelSelector::Second => MODEL2.lock().await,
        ModelSelector::HydeFirst => HYDE_MODEL1.lock().await,
        ModelSelector::HydeSecond => HYDE_MODEL2.lock().await,
    };
    
    let tokenizer = loaded_model.1.clone();
    let device = loaded_model.2.clone();
    let max_seq_len = loaded_model.3;
    let model = &mut loaded_model.0;

    let mut response_chunks = vec![];
    let mut tos = TokenOutputStream::new(tokenizer);
   
    flush_message("Tokenizing prompt...", &mut websocket, FlushType::Status).await?;
    // Parse the prompt to a raw string format.
    let prompt_str = parse_prompt_to_raw(prompt);
    // Tokenize the prompt string for model processing.
    let tokens = tos
        .tokenizer()
        .encode(prompt_str, true)
        .map_err(anyhow::Error::msg)?;
    
    // Optionally, print each token and its ID if verbose logging is enabled.
    if VERBOSE_PROMPT {
        for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
            let token = token.replace('▁', " ").replace("<0x0A>", "\n");
            debug!("{id:7} -> '{token}'");
        }
    }
    
    // Handle token length restrictions by trimming if necessary.
    flush_message("Clipping prompt...", &mut websocket, FlushType::Status).await?;
    let prompt_tokens = tokens.get_ids();
    let to_sample = SAMPLE_LEN.saturating_sub(1);
    
    let prompt_tokens = if prompt_tokens.len() + to_sample > max_seq_len - 10 {
        let to_remove = prompt_tokens.len() + to_sample + 10 - max_seq_len;
        prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
    } else {
        prompt_tokens.to_vec()
    };
    
    // Setup for generating model responses.
    match hyde {
        true => flush_message("Figuring out question...", &mut websocket, FlushType::Status).await?,
        false => flush_message("Generating tokens...", &mut websocket, FlushType::Status).await?,
    }
    
    let mut all_tokens = vec![];
    let mut logits_processor = setup_logit_procesing();

    let start_prompt_processing: std::time::Instant = std::time::Instant::now();
    let mut next_token = if !SPLIT_PROPMT {
        // Generate response in a single batch if not splitting.
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        logits_processor.sample(&logits)?
    } else {
        // Generate response token by token if splitting.
        let mut next_token = 0;
        for (pos, token) in prompt_tokens.iter().enumerate() {
            let input = Tensor::new(&[*token], &device)?.unsqueeze(0)?;
            let logits = model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            next_token = logits_processor.sample(&logits)?
        }
        next_token
    };
    let prompt_dt = start_prompt_processing.elapsed();
    all_tokens.push(next_token);
    
    
    // Collect chunks of the generated response.
    if let Some(token) = tos.next_token(next_token)? {
        if !hyde {
            let _ = flush_message(&token, &mut websocket, FlushType::Token).await;
        }
        response_chunks.push(token);
    }

    // Continue generating tokens until the sample length is reached or an end-of-sentence token is encountered.
    let eos_token = get_eos_token(hyde);
    let eos_token = *tos.tokenizer().get_vocab(true).get(&eos_token).unwrap();
    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    for index in 0..to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if REPEAT_PENALTY == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(REPEAT_LAST_N);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                REPEAT_PENALTY,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        if let Some(token) = tos.next_token(next_token)? {
            if !hyde {
                let _ = flush_message(&token, &mut websocket, FlushType::Token).await;
            }
            response_chunks.push(token);
        }
        sampled += 1;
        if next_token == eos_token {
            break;
        };
    }
    
    let dt = start_post_prompt.elapsed();
    if VERBOSE_PROMPT {
        // Optionally print the final output and performance stats if verbose logging is enabled.
        if let Some(rest) = tos.decode_rest().map_err(Error::msg)? {
            if !hyde {
                let _ = flush_message(&rest, &mut websocket, FlushType::Token).await;
            }
            response_chunks.push(rest);
        }
        std::io::stdout().flush()?;
    
        info!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        info!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );
    }

    if !hyde {
        flush_message("Done!", &mut websocket, FlushType::Status).await?;
    }
    
    Ok(response_chunks.join(""))
}
