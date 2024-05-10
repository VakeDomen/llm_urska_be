use std::env;

use once_cell::sync::Lazy;

use crate::{controllers::collector::Collector, llm::{loader::LoadType, model::ModelArchitecture}};

// WSS Config
pub const LISTEN_ADDRESS: &str = "127.0.0.1:6668";
pub const QDRANT_SERVER: &str = "http://localhost:6334";
pub const QDRANT_COLLECTION: &str = "urska_bge_m3_decorated";
pub static MYSQL_URL: Lazy<String> =  Lazy::new(|| env::var("DATABASE_URL").unwrap_or("mysql://user:pw@localhost/db".to_string()));

pub const USE_HYDE: bool = true;

pub const COLLECOTRS: Lazy<Vec<Collector>> = Lazy::new(|| vec![
    Collector::Rag("urska_m3_shard_bachelor".into(), 3),
    Collector::Rag("urska_m3_shard_masters".into(), 3),
    Collector::Rag("urska_m3_shard_phd".into(), 3),
    Collector::Rag("urska_m3_shard_general".into(), 3),
    Collector::Raw("urska_m3_shard_staff".into(), 3),
    Collector::Rag("urska_m3_shard_erasmus".into(), 3),
]);

// LLM config
pub const SYSTEM_MSG: &str = "Your name is Urška. You are a chatbot asistant at University of primorska, more specific FAMNIT. Your task is to help students in the name of student's office. Answer the student's question in the name of the student office with factual data. ";
pub const SYSTEM_RAG_MSG: &str = "Your name is Urška. You are a chatbot asistant at University of primorska, more specific FAMNIT. Your task is to help students in the name of student's office. Answer the student's question in the name of the student office with factual data. You are also given passages from out internal documentation to help you derive the answer. Answer based on the documentation and not your prior knowledge but don't mention the passages in the funal answer.";
pub const HYDE_SYSTEM_MSG: &str = "Your task is to generate a short markdown passage that should resable a passage that could be found in information of University of Primorska - Faculty of Mathematics, Natural Sciences and Information Technologies that is ment to answer stuent questions. In terms of markdown you are allowed to use headings and lists and tables.";

pub const MODEL_ARCITECTURE: ModelArchitecture = ModelArchitecture::Mixtral;
pub const MODEL_PATH: &str = "models/mixtral8x7b/mixtral-8x7b-instuct-v0.1Q5_K_M.gguf";
pub const TOKENIZER_PATH: &str = "models/mixtral8x7b/tokenizer.json";

pub const HYDE_MODEL_ARCITECTURE: ModelArchitecture = ModelArchitecture::Llama3;
pub const HYDE_MODEL_PATH: &str = "models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf";
pub const HYDE_TOKENIZER_PATH: &str = "models/llama3-8b/tokenizer.json";


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