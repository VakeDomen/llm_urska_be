pub const DEFAULT_PROMPT: &str = "Kako naj dobim čim več všečkov na svojih socjalnih omrežjih?";
pub const SYSTEM_MSG: &str = "Si vplivnež pomagaj mi! Odgovarjaj v Slovenščini!";

pub const SEED: u64 = 42;

pub const TEMPERATURE: f64 = 0.4;
pub const SAMPLE_LEN: usize = 500;
pub const TOP_K: Option<usize> = Some(5);
pub const TOP_P: Option<f64> = None;

pub const VERBOSE_PROMPT: bool = true;
pub const SPLIT_PROPMT: bool = true;
pub const REPEAT_PENALTY: f32 = 1.1;
pub const REPEAT_LAST_N: usize = 64;