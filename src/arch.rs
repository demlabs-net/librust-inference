//! Реестр архитектур (A1 плана «архитектуры llama.cpp + тиринг»): что уже
//! поддержано, что планируется и что осознанно вне плана.
//!
//! Зачем: движок читал конфиг по префиксу `<arch>.…` и на чужой архитектуре
//! падал «missing blk.0.attn_q.weight» — по такой ошибке нельзя понять, это
//! битый файл или просто незнакомая модель. Теперь у каждой архитектуры есть
//! явный статус, а загрузка называет архитектуру и список поддержанных.
//!
//! Список имён — из llama.cpp нашего эталонного коммита (`LLM_ARCH_*` в
//! `src/llama-arch.h`, 149 записей). Полнота списка проверяется тестом.

use crate::error::BackendError;

/// Семейство (для приоритизации волн в плане и для «что дальше»).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// Классический плотный трансформер (llama/mistral/gemma/phi/…).
    Dense,
    /// MoE-роутер + эксперты.
    Moe,
    /// Гибрид: рекуррентные миксеры (GDN/SSM/Mamba/RWKV) + внимание.
    Hybrid,
    /// Разреженное внимание с индексером (GLM-DSA, Qwen3.8/QSA-like).
    Sparse,
    /// Мультимодальный (vision/audio-энкодер в той же модели).
    Multimodal,
    /// Энкодер (эмбеддинги/BERT-семейство).
    Encoder,
    /// Аудио/TTS-кодеки (у нас — speech-крейт, не эта ветка).
    Audio,
    /// Прочее/не классифицировано.
    Other,
}

/// Статус поддержки у нас.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// Работает, есть text_parity с llama.cpp.
    Supported,
    /// В плане (волна A2-A6), ещё нет.
    Planned,
    /// Осознанно не берём (см. план: аудио-кодеки, OCR сверх нужного).
    OutOfScope,
}

/// Описание архитектуры.
#[derive(Debug, Clone, Copy)]
pub struct ArchInfo {
    /// Имя как в `general.architecture` GGUF.
    pub name: &'static str,
    pub family: Family,
    pub status: Support,
    /// Что это и чем отличается (для сообщений об ошибках и панели).
    pub note: &'static str,
}

const fn a(name: &'static str, family: Family, status: Support, note: &'static str) -> ArchInfo {
    ArchInfo {
        name,
        family,
        status,
        note,
    }
}

/// Архитектуры, которые у нас реализованы (или их текстовое ядро).
pub const SUPPORTED: &[ArchInfo] = &[
    a("qwen3", Family::Dense, Support::Supported, "Qwen3 (dense); текстовое ядро также для qwen3vl/omni"),
    a("qwen35", Family::Hybrid, Support::Supported, "Qwen3.5/3.8: GDN-миксер + внимание"),
    a("qwen3vl", Family::Multimodal, Support::Supported, "Qwen3-VL: текстовое ядро + vision-энкодер"),
    a("qwen3vlmoe", Family::Multimodal, Support::Planned, "Qwen3-VL-MoE: нужен MoE-каркас (A3)"),
    a("dflash", Family::Other, Support::Supported, "DFlash-драфтер (спек-декодинг)"),
    a("bert", Family::Encoder, Support::Supported, "BERT-эмбеддинги (bge)"),
];

/// Полный список имён llama.cpp (149; `src/llama-arch.h`, коммит 5266f24da).
pub const LLAMACPP_ARCHS: &[&str] = &[
    "clip", "llama", "llama4", "deci", "falcon",
    "grok", "gpt2", "gptj", "gptneox", "mpt",
    "baichuan", "starcoder", "refact", "bert", "modern-bert",
    "nomic-bert", "nomic-bert-moe", "neo-bert", "jina-bert-v2", "jina-bert-v3",
    "eurobert", "bloom", "stablelm", "qwen", "qwen2",
    "qwen2moe", "qwen2vl", "qwen3", "qwen3moe", "qwen3next",
    "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "qwen4exp",
    "phi2", "phi3", "phimoe", "plamo", "plamo2",
    "plamo3", "codeshell", "orion", "internlm2", "minicpm",
    "minicpm3", "gemma", "gemma2", "gemma3", "gemma3n",
    "gemma4", "gemma4-assistant", "gemma-embedding", "starcoder2", "mamba",
    "mamba2", "jamba", "falcon-h1", "xverse", "command-r",
    "cohere2", "cohere2moe", "dbrx", "olmo", "olmo2",
    "olmoe", "muse-glimmer", "openelm", "arctic", "deepseek",
    "deepseek2", "deepseek2-ocr", "deepseek32", "deepseek4", "chatglm",
    "glm4", "glm4moe", "glm-dsa", "bitnet", "t5",
    "t5encoder", "jais", "jais2", "nemotron", "nemotron_h",
    "nemotron_h_moe", "exaone", "exaone4", "exaone-moe", "rwkv6",
    "rwkv6qwen2", "rwkv7", "arwkv7", "granite", "granitemoe",
    "granitehybrid", "graniteswitch", "granite_swa", "chameleon", "wavtokenizer-dec",
    "plm", "bailingmoe", "bailingmoe2", "bailingmoe3", "dots1",
    "dots3note", "arcee", "afmoe", "laguna", "ernie4_5",
    "ernie4_5-moe", "hunyuan-moe", "hunyuan-dense", "hunyuan_vl", "hy_v3",
    "smollm3", "gpt-oss", "lfm2", "lfm2moe", "dream",
    "smallthinker", "llada", "llada-moe", "seed_oss", "grovemoe",
    "apertus", "minimax-01", "minimax-m2", "minimax-m3", "cogvlm",
    "rnd1", "pangu-embedded", "mistral3", "eagle3", "dflash",
    "mistral4", "paddleocr", "mimo2", "step35", "llama-embed",
    "maincoder", "kimi-linear", "kimi-k3", "talkie", "mellum",
    "nanbeige", "qwen3tts", "pockettts", "(unknown)",
];

/// Известные архитектуры, которых у нас нет (имя, семейство, примечание) —
/// используется в сообщениях об ошибке и панелью («что не поддерживается»).
const PLANNED: &[ArchInfo] = &[
    a("glm4", Family::Dense, Support::Planned, "GLM-4 (частичный RoPE, sandwich-нормы) — волна A2"),
    a("glm4moe", Family::Moe, Support::Planned, "GLM-4.5/4.6 MoE — A2 после MoE-каркаса"),
    a("glm-dsa", Family::Sparse, Support::Planned, "GLM-DSA: разреженное внимание с индексером"),
    a("chatglm", Family::Dense, Support::Planned, "легаси ChatGLM (fused qkv/up)"),
    a("qwen3next", Family::Sparse, Support::Planned, "Qwen3-Next/3.8-Flash: GDN+индексер+MoE+PLE — A5"),
    a("qwen3moe", Family::Moe, Support::Planned, "Qwen3-MoE — волна A3 (MoE-каркас)"),
    a("qwen35moe", Family::Moe, Support::Planned, "Qwen3.5-MoE — A3"),
    a("llama", Family::Dense, Support::Planned, "Llama 1/2/3 — A6"),
    a("mistral3", Family::Dense, Support::Planned, "Mistral/Ministral — A6"),
    a("gemma2", Family::Dense, Support::Planned, "Gemma 2/3 — A6"),
    a("deepseek2", Family::Dense, Support::Planned, "DeepSeek MLA — A6"),
];

/// Найти архитектуру: сначала поддержанные, затем запланированные.
pub fn lookup(name: &str) -> Option<ArchInfo> {
    SUPPORTED
        .iter()
        .chain(PLANNED.iter())
        .find(|i| i.name == name)
        .copied()
}

/// Поддержана ли архитектура (текстовым ядром).
pub fn is_supported(name: &str) -> bool {
    SUPPORTED.iter().any(|i| i.name == name)
}

/// Имена поддержанных архитектур (для сообщений и README).
pub fn supported_names() -> Vec<&'static str> {
    SUPPORTED.iter().map(|i| i.name).collect()
}

/// Внятная ошибка «эта архитектура не поддержана» (вместо «missing tensor»).
pub fn unsupported_arch(arch: &str) -> BackendError {
    let list = supported_names().join(", ");
    match lookup(arch) {
        Some(info) => BackendError::UnsupportedFormat(format!(
            "архитектура {arch} ({:?}) пока не поддержана: {}. \
             Поддержаны: {list}. План волн — .zcode/plans/plan-archs-and-tiered-inference.md",
            info.family, info.note
        )),
        None => BackendError::UnsupportedFormat(format!(
            "архитектура {arch} не поддержана (нет и в llama.cpp-списке из 149 имён). \
             Поддержаны: {list}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Имена в списке llama.cpp уникальны и не пусты, а каждая поддержанная
    /// архитектура в нём есть (иначе мы «поддерживаем» то, чего нет).
    #[test]
    fn inventory_is_consistent() {
        let mut seen = std::collections::HashSet::new();
        for n in LLAMACPP_ARCHS {
            assert!(!n.is_empty());
            assert!(seen.insert(*n), "дубликат архитектуры: {n}");
        }
        assert_eq!(seen.len(), 149, "в llama.cpp 5266f24da ровно 149 строк LLM_ARCH_*");
        for info in SUPPORTED {
            assert!(
                seen.contains(info.name),
                "{} отсутствует в списке llama.cpp",
                info.name
            );
        }
    }

    /// Сообщение об ошибке называет архитектуру и список поддержанных.
    #[test]
    fn unsupported_error_is_actionable() {
        let e = unsupported_arch("glm4").to_string();
        assert!(e.contains("glm4"), "{e}");
        assert!(e.contains("qwen3"), "{e}");
        let e = unsupported_arch("вася-нет-такой").to_string();
        assert!(e.contains("вася-нет-такой") && e.contains("qwen3"), "{e}");
    }

    #[test]
    fn lookup_finds_planned_and_supported() {
        assert!(is_supported("qwen3") && is_supported("qwen35"));
        assert!(!is_supported("glm4"));
        assert_eq!(lookup("glm4").map(|i| i.family), Some(Family::Dense));
        assert_eq!(lookup("glm4moe").map(|i| i.status), Some(Support::Planned));
        assert!(lookup("нет-такой").is_none());
    }
}
