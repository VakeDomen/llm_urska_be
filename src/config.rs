pub const DEFAULT_PROMPT: &str = "What is your name and your purpose?";
pub const SYSTEM_MSG: &str = "You are a student assistant named Urška. Help the student with his question:";

pub const SEED: u64 = 42;

pub const TEMPERATURE: f64 = 0.1;
pub const SAMPLE_LEN: usize = 500;
pub const TOP_K: Option<usize> = None;
pub const TOP_P: Option<f64> = None;

pub const VERBOSE_PROMPT: bool = true;
pub const SPLIT_PROPMT: bool = true;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;
