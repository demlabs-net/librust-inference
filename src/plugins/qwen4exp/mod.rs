//! Плагин `qwen4exp` — Qwen3.8-Flash-Next.
//!
//! Один каталог на архитектуру: конфиг и схема файла ([`config`]),
//! hyper-connections ([`hyper`]), индексер ([`qsa`]) и PLE ([`ple`]);
//! загрузчик весов из контейнера NINFER придёт сюда же (`load.rs`), чтобы
//! forward и чтение весов не расползались по крейту.
//!
//! Статус (11.11.2026): forward в работе. Модули слоёв готовы и покрыты
//! тестами (mix/combine HC, relu-скоринг и top-k индексера, n-граммные
//! строки PLE, общий эксперт MoE), чего ещё нет — сборки слоя и загрузчика,
//! поэтому `implemented = false` и загрузчик отказывает внятным текстом
//! через [`EnginePlugin::planned_error`].

pub mod config;
pub mod hyper;
pub mod ple;
pub mod qsa;

use std::borrow::Cow;

use crate::arch::Family;
use crate::error::BackendError;
use crate::plugin::{
    Capabilities, EnginePlugin, ModelVersion, Offload, PluginKind, SpecBackend, Version,
};

/// Архитектура модели в GGUF/манифесте.
pub const ARCH: &str = config::ARCH;

/// Единственная модельная версия плагина.
pub const VERSION: ModelVersion = ModelVersion {
    name: "v4exp",
    arch: ARCH,
    implemented: false,
    plan: Some("inference/docs/plugin-platform-plan.md (Э2)"),
};

/// Версия самого плагина: 0.1.0 — модули слоёв готовы, сборки модели нет.
pub const PLUGIN_VERSION: Version = Version::new(0, 1, 0);

const CAPS: Capabilities = Capabilities {
    family: Family::Moe,
    implemented: false,
    // KV-форматы появятся вместе с forward; обещать заранее нельзя.
    kv_formats: &[],
    speculative: &[SpecBackend::Off],
    // 46.9 GiB экспертов в 32 ГБ VRAM не живут — offload обязателен.
    offload: Offload {
        experts: true,
        kv: false,
        prefix_cache: false,
    },
    batch_slots: 2,
    max_context: 262_144,
    note: "Qwen3.8-Flash-Next: MoE 512/top-10 + shared, hyper-connections, \
           QSA-индексер, PLE; веса — контейнер NINFER, forward в работе (Э2)",
};

/// Плагин Flash-Next.
#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen4ExpPlugin;

impl EnginePlugin for Qwen4ExpPlugin {
    fn id(&self) -> &str {
        "qwen4exp"
    }

    fn kind(&self) -> PluginKind {
        PluginKind::Native
    }

    fn version(&self) -> Version {
        PLUGIN_VERSION
    }

    fn model_versions(&self) -> Cow<'static, [ModelVersion]> {
        Cow::Borrowed(&[VERSION])
    }

    fn capabilities(&self, v: &ModelVersion) -> Capabilities {
        if v.arch == ARCH {
            CAPS
        } else {
            Capabilities::unknown("плагин qwen4exp обслуживает только qwen4exp")
        }
    }

    /// Отказ с указанием, что именно уже готово: иначе пользователь видит
    /// родовое «архитектура в плане» и не понимает, сколько осталось.
    fn planned_error(&self, v: &ModelVersion) -> Option<BackendError> {
        if v.implemented {
            return None;
        }
        Some(BackendError::UnsupportedFormat(format!(
            "qwen4exp ({ARCH}): forward ещё собирается (Э2). Готово: общий эксперт MoE, \
             hyper-connections (mix/combine), индексер (сжатие/relu/top-k), PLE-хэши, \
             читатель весов NINFER. Осталось: загрузчик модели, слой GDN/attention + HC + \
             MoE + PLE, parity с Infernix"
        )))
    }
}

/// Статический экземпляр для реестра.
pub static PLUGIN: Qwen4ExpPlugin = Qwen4ExpPlugin;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declares_flash_next_as_planned() {
        let p = &PLUGIN;
        assert_eq!(p.id(), "qwen4exp");
        assert!(matches!(p.kind(), PluginKind::Native));
        let versions = p.model_versions();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].arch, ARCH);
        assert!(!versions[0].implemented);
        // способности честные: MoE, offload экспертов обязателен
        let caps = p.capabilities(&versions[0]);
        assert!(matches!(caps.family, Family::Moe));
        assert!(caps.offload.experts);
        assert!(!caps.implemented);
    }

    /// Отказ говорит, что уже готово и что осталось — а не «missing tensor».
    #[test]
    fn planned_error_is_actionable() {
        let e = PLUGIN.planned_error(&VERSION).expect("отказ для плана");
        let text = format!("{e}");
        assert!(text.contains("hyper-connections"), "{text}");
        assert!(text.contains("NINFER"), "{text}");
        assert!(!text.contains("missing tensor"), "{text}");
        // у реализованной версии отказа нет
        let done = ModelVersion {
            implemented: true,
            ..VERSION
        };
        assert!(PLUGIN.planned_error(&done).is_none());
    }

    #[test]
    fn foreign_arch_gets_unknown_caps() {
        let other = ModelVersion {
            name: "v3",
            arch: "qwen3",
            implemented: true,
            plan: None,
        };
        let caps = PLUGIN.capabilities(&other);
        assert_eq!(
            caps.batch_slots, 0,
            "чужая архитектура — нулевые способности"
        );
    }
}
