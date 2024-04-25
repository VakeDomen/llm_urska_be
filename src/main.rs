use std::io::Write;

use anyhow::{Error, Result};
use candle_core::{self, quantized::gguf_file::Content, Device};
use tokenizers::Tokenizer;
use candle_transformers::models::quantized_llama as model;
use model::ModelWeights;

const DEFAULT_PROMPT: &str = "How do i enroll into next year CS?";
const SYSTEM_MSG: &str = "You are a student assistant named Urška. Help the student with his question:";
const VERBOSE_PROMPT: bool = true;
const SAMPLE_LEN: usize = 1000;

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

    match prompt_model(model, tokenizer, prompt) {
        Ok(out) => println!("{}", out),
        Err(e) => panic!("Can't prompt model: {:#?}", e),
    }
    

}


fn prompt_model(model: ModelWeights, tokenizer: Tokenizer, prompt: Prompt) -> Result<String> {
    // let mut pre_prompt_tokens = vec![];
    let mut tos = TokenOutputStream::new(tokenizer);
   
//    for prompt_index in 0.. {
        let prompt_str = parse_prompt_to_raw(prompt)?;
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

        
    // }

    Ok("HEY".to_string())
}

fn parse_prompt_to_raw(prompt: Prompt) -> Result<String> {
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