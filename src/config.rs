use crate::llm::{loader::LoadType, model::ModelArchitecture};

// WSS Config
pub const LISTEN_ADDRESS: &str = "127.0.0.1:6668";
pub const QDRANT_SERVER: &str = "http://localhost:6334";
pub const QDRANT_COLLECTION: &str = "urska_bge_m3_decorated";

// LLM config
pub const SYSTEM_MSG: &str = "Your name is Urška. You are a chatbot asistant at University of primorska, more specific FAMNIT. Your task is to help students in the name of student's office. Answer the student's question in the name of the student office with factual data. ";
pub const SYSTEM_RAG_MSG: &str = "Your name is Urška. You are a chatbot asistant at University of primorska, more specific FAMNIT. Your task is to help students in the name of student's office. Answer the student's question in the name of the student office with factual data. You are also given passages from out internal documentation to help you derive the answer. Answer based on the documentation and not your prior knowledge but don't mention the passages in the funal answer.";

pub const MODEL_ARCITECTURE: ModelArchitecture = ModelArchitecture::Mixtral;
// pub const MODEL_PATH: &str = "models/llama3-70b/Meta-Llama-3-70B-Instruct.Q4_K_S.gguf";
// pub const TOKENIZER_PATH: &str = "models/llama3-70b/tokenizer.json";

// pub const MODEL_PATH: &str = "models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf";
// pub const TOKENIZER_PATH: &str = "models/llama3-8b/tokenizer.json";

pub const MODEL_PATH: &str = "models/mixtral8x7b/mixtral-8x7b-instuct-v0.1Q5_K_M.gguf";
pub const TOKENIZER_PATH: &str = "models/mixtral8x7b/tokenizer.json";


pub const EMBEDDING_MODEL_TYPE: LoadType = LoadType::PyBin;
pub const EMBEDDING_MODEL_PATH: &str = "../llm_urska_be/models/bge-m3";
pub const EMBEDDING_TOKENIZER: &str = "../llm_urska_be/models/bge-m3/tokenizer.json";

pub const SEED: u64 = 42;

pub const TEMPERATURE: f64 = 0.4;
pub const SAMPLE_LEN: usize = 1000;
pub const TOP_K: Option<usize> = None;
pub const TOP_P: Option<f64> = None;

pub const VERBOSE_PROMPT: bool = true;
pub const SPLIT_PROPMT: bool = false;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;