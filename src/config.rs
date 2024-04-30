// WSS Config
pub const LISTEN_ADDRESS: &str = "127.0.0.1:4321";
pub const QDRANT_SERVER: &str = "http://localhost:6334";
pub const QDRANT_COLLECTION: &str = "baai-large-en-ft-web";

// LLM config
pub const SYSTEM_MSG: &str = "Your name is Urška. You are a chatbot asistant at University of primorska, more specific FAMNIT. Your task is to help students in the name of student's office. Answer the student's question in the name of the student office with factual data. ";
pub const SYSTEM_RAG_MSG: &str = "Your name is Urška. You are a chatbot asistant at University of primorska, more specific FAMNIT. Your task is to help students in the name of student's office. Answer the student's question in the name of the student office with factual data. You are also given passages from out internal documentation to help you derive the answer. Answer based on the documentation and not your prior knowledge but don't mention the passages in the funal answer.";

pub const MODEL_PATH: &str = "models/llama3-8b/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf";
pub const EMBEDDING_MODEL_PATH: &str = "models/bge-large-en-v1.5-ft";
pub const SEED: u64 = 42;

pub const TEMPERATURE: f64 = 0.4;
pub const SAMPLE_LEN: usize = 1000;
pub const TOP_K: Option<usize> = None;
pub const TOP_P: Option<f64> = None;

pub const VERBOSE_PROMPT: bool = true;
pub const SPLIT_PROPMT: bool = true;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;