use std::io::Write;

use anyhow::{Error, Result};
use candle_core::{self, quantized::gguf_file::Content, Device, Tensor};
use tokenizers::Tokenizer;
use candle_transformers::{generation::{LogitsProcessor, Sampling}, models::quantized_llama as model};
use model::ModelWeights;

const DEFAULT_PROMPT: &str = "What is your name and your purpose?";
const SYSTEM_MSG: &str = "You are a student assistant named Urška. Help the student with his question:";

const SEED: u64 = 42;

const TEMPERATURE: f64 = 0.1;
const SAMPLE_LEN: usize = 500;
const TOP_K: Option<usize> = None;
const TOP_P: Option<f64> = None;

const VERBOSE_PROMPT: bool = true;
const SPLIT_PROPMT: bool = false;
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 64;

#[derive(Debug)]
enum Prompt {
    Interactive,
    Chat,
    One(String),
}


/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}


fn main() {
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle_core::utils::with_avx(),
        candle_core::utils::with_neon(),
        candle_core::utils::with_simd128(),
        candle_core::utils::with_f16c()
    );

    let device = match Device::new_cuda(0) {
        Ok(cuda) => cuda,
        Err(e) => {
            println!("Error initializing CUDA device. Switching to CPU.");
            Device::Cpu
        },
    };

    let tokenizer = match tokenizer("models/llama3-8b/tokenizer.json") {
        Ok(t) => t,
        Err(e) => panic!("Can't load tokenizer: {:#?}", e),
    };

    let model = match  model("models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf", &device) { 
        Ok(m) => m,
        Err(e) => panic!("Can't load model: {:#?}", e),
    };

    let prompt = Prompt::One(DEFAULT_PROMPT.to_string());

    match prompt_model(model, tokenizer, prompt, &device) {
        Ok(out) => println!("{}", out),
        Err(e) => panic!("Can't prompt model: {:#?}", e),
    }
    

}


fn prompt_model(mut model: ModelWeights, tokenizer: Tokenizer, prompt: Prompt, device: &Device) -> Result<String> {
    let mut pre_prompt_tokens = vec![];
    let mut tos = TokenOutputStream::new(tokenizer);
   
   for prompt_index in 0.. {
        let prompt_str = parse_prompt_to_raw(&prompt)?;
        print!("{}", &prompt_str);
        let tokens = tos
            .tokenizer()
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;
        
        if VERBOSE_PROMPT {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();
        let to_sample = SAMPLE_LEN.saturating_sub(1);
        
        let prompt_tokens = if prompt_tokens.len() + to_sample > model::MAX_SEQ_LEN - 10 {
            let to_remove = prompt_tokens.len() + to_sample + 10 - model::MAX_SEQ_LEN;
            prompt_tokens[prompt_tokens.len().saturating_sub(to_remove)..].to_vec()
        } else {
            prompt_tokens
        };
        
        let mut all_tokens = vec![];
        let mut logits_processor = {
            
            let sampling = if TEMPERATURE <= 0. {
                Sampling::ArgMax
            } else {
                match (TOP_K, TOP_P) {
                    (None, None) => Sampling::All { temperature: TEMPERATURE },
                    (Some(k), None) => Sampling::TopK { k, temperature: TEMPERATURE },
                    (None, Some(p)) => Sampling::TopP { p, temperature: TEMPERATURE },
                    (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p,temperature:  TEMPERATURE },
                }
            };
            LogitsProcessor::from_sampling(SEED, sampling)
        };

        let start_prompt_processing = std::time::Instant::now();
        let mut next_token = if !SPLIT_PROPMT {
            let input = Tensor::new(prompt_tokens.as_slice(), device)?.unsqueeze(0)?;
            let logits = model.forward(&input, 0)?;
            let logits = logits.squeeze(0)?;
            logits_processor.sample(&logits)?
        } else {
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
        if let Some(t) = tos.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        let eos_token = "<|end_of_text|>";
        let eos_token = *tos.tokenizer().get_vocab(true).get(eos_token).unwrap();
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
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            sampled += 1;
            if next_token == eos_token {
                break;
            };
        }
        if let Some(rest) = tos.decode_rest().map_err(Error::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        let dt = start_post_prompt.elapsed();
        println!(
            "\n\n{:4} prompt tokens processed: {:.2} token/s",
            prompt_tokens.len(),
            prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
        );
        println!(
            "{sampled:4} tokens generated: {:.2} token/s",
            sampled as f64 / dt.as_secs_f64(),
        );

        match prompt {
            Prompt::One(_) => break,
            Prompt::Interactive => {}
            Prompt::Chat => {
                pre_prompt_tokens = [prompt_tokens.as_slice(), all_tokens.as_slice()].concat()
            }
        }
        
    }

    Ok("DONE".to_string())
}

fn parse_prompt_to_raw(prompt: &Prompt) -> Result<String> {
    match &prompt {
        Prompt::One(prompt) => Ok(llama3_prompt(prompt.clone())),
        Prompt::Interactive | Prompt::Chat => {
            let is_interactive = matches!(prompt, Prompt::Interactive);
            print!("> ");
            std::io::stdout().flush()?;
            let mut prompt = String::new();
            std::io::stdin().read_line(&mut prompt)?;
            if prompt.ends_with('\n') {
                prompt.pop();
                if prompt.ends_with('\r') {
                    prompt.pop();
                }
            }
            Ok(llama3_prompt(prompt))
        }
    }
}

fn llama3_prompt(user_msg: String) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
        SYSTEM_MSG,
        user_msg
    )
}

fn tokenizer(tokenizer_path: &str) -> Result<Tokenizer> {
    let tokenizer_path = std::path::PathBuf::from(tokenizer_path);
    Tokenizer::from_file(tokenizer_path).map_err(Error::msg)
}

fn model(model_path: &str, device: &Device) -> Result<ModelWeights> {
    let model_path = std::path::PathBuf::from(model_path);
    let mut file = std::fs::File::open(&model_path)?;
    let start = std::time::Instant::now();
    
    let model: Content = Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    
    println!(
        "Loaded {:?} tensors in {:.2}s",
        model.tensor_infos.len(),
        start.elapsed().as_secs_f32(),
    );
    
    let weights = ModelWeights::from_gguf(model, &mut file, device)?;
    Ok(weights)
}



impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => panic!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}