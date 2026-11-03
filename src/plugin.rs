//! Плагинный слой движков инференса.
//!
//! Каждый движок — плагин: `Native` (компилируется в процесс: qwen,
//! speech, …) или `External` (адаптер над HTTP/IPC-эндпоинтом: Infernix,
//! Basalt, Strata, vLLM, LM Studio, MLX-VLM, HyperQwen, …). Сам крейт
//! (`rust-inference`) — общая для всех плагинов библиотека инструментов:
//! контракт [`Backend`](crate::backend::Backend), GGUF-кодеки и кванты,
//! токенизатор и чат-шаблоны, сэмплеры, KV-форматы, реестр архитектур.
//!
//! Версионирование модели внутри движка — отдельная ось от версии самого
//! плагина: `QwenPlugin` (своя semver) обслуживает **модельные версии**
//! `v3` (dense), `v3.5` (гибрид GDN, включая Qwen3.8-27B) и `v4exp`
//! (Qwen3.8-Flash-Next, в плане). Способности (`Capabilities`) объявляются
//! на каждую модельную версию, чтобы сервис и балансировщик маршрутизировали
//! по факту, а не по имени движка.

use std::fmt;
use std::str::FromStr;

use crate::arch::{Family, Support};
use crate::dtype::KvFormat;
use crate::error::BackendError;

/// Semver плагина (и внешних движков, когда те её сообщают).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl Version {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for Version {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut parts = s.split('.');
        let take = |p: Option<&str>| -> Result<u32, String> {
            let raw = p.ok_or_else(|| format!("версия «{s}» не в формате X.Y.Z"))?;
            raw.parse::<u32>()
                .map_err(|_| format!("версия «{s}» не в формате X.Y.Z"))
        };
        let major = take(parts.next())?;
        let minor = take(parts.next())?;
        let patch = take(parts.next())?;
        if parts.next().is_some() {
            return Err(format!("версия «{s}» не в формате X.Y.Z"));
        }
        Ok(Self { major, minor, patch })
    }
}

/// Спекулятивный бэкенд декодинга.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpecBackend {
    /// Чистый авторегрессивный декод.
    Off,
    /// MTP-голова самой модели.
    Mtp,
    /// DFlash v1 (диффузия блоков).
    DFlash,
    /// DFlash v2 (селектор/conv).
    DFlash2,
    /// n-gram prompt lookup по истории (без драфтера).
    NGramLookup,
}

/// Что движок умеет выносить из VRAM (тиринг памяти).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Offload {
    /// Эксперты MoE в RAM/SSD с кэшем в VRAM.
    pub experts: bool,
    /// KV-кэш, стримируемый из RAM (resident-окно).
    pub kv: bool,
    /// Блочный prefix cache с тирами (VRAM/RAM/файл).
    pub prefix_cache: bool,
}

impl Offload {
    pub const NONE: Self = Self { experts: false, kv: false, prefix_cache: false };
}

/// Модельная версия, которую обслуживает плагин: публичное имя (`v3`,
/// `v3.5`, `v4exp`), GGUF-архитектура и статус реализации.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelVersion {
    pub name: &'static str,
    pub arch: &'static str,
    pub implemented: bool,
    /// Указатель на план (документ/волну), когда `implemented == false`.
    pub plan: Option<&'static str>,
}

impl ModelVersion {
    pub const fn new(name: &'static str, arch: &'static str, implemented: bool) -> Self {
        Self { name, arch, implemented, plan: None }
    }

    pub const fn planned(name: &'static str, arch: &'static str, plan: &'static str) -> Self {
        Self { name, arch, implemented: false, plan: Some(plan) }
    }
}

/// Способности модельной версии: по ним сервис/балансировщик решают, кому
/// отдать запрос, не спрашивая сам движок.
#[derive(Debug, Clone, Copy)]
pub struct Capabilities {
    pub family: Family,
    pub implemented: bool,
    pub kv_formats: &'static [KvFormat],
    pub speculative: &'static [SpecBackend],
    pub offload: Offload,
    /// Максимум одновременных запросов (слоты).
    pub batch_slots: u32,
    /// Максимальный контекст, токенов.
    pub max_context: u32,
    /// Человекочитаемая заметка (архитектурная сводка).
    pub note: &'static str,
}

impl Capabilities {
    /// Сводка из реестра архитектур: семейство и статус берутся из
    /// [`crate::arch`], остальное — аргументами (у каждой версии свои
    /// кернелы, KV и спека).
    pub const fn from_arch(
        arch: &'static str,
        kv_formats: &'static [KvFormat],
        speculative: &'static [SpecBackend],
        offload: Offload,
        batch_slots: u32,
        max_context: u32,
        note: &'static str,
    ) -> Self {
        // `arch::lookup` не const, поэтому семейство подставляется
        // вызывающим через `family_for_arch` ниже; пустые списки означают
        // «ещё не определено» (планируемая версия).
        let _ = arch;
        Self {
            family: Family::Other,
            implemented: false,
            kv_formats,
            speculative,
            offload,
            batch_slots,
            max_context,
            note,
        }
    }

    /// Семейство по GGUF-архитектуре (через реестр; unknown → `Other`).
    pub fn family_for_arch(arch: &str) -> Family {
        crate::arch::lookup(arch).map(|i| i.family).unwrap_or(Family::Other)
    }
}

/// Как плагин живёт в процессе.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PluginKind {
    /// Компилируется в процесс (qwen, speech, …).
    Native,
    /// Адаптер над внешним движком по HTTP/IPC.
    External,
}

/// Контракт движка-плагина.
pub trait EnginePlugin: Send + Sync {
    fn id(&self) -> &str;
    fn kind(&self) -> PluginKind;
    /// Версия самого плагина (не модели).
    fn version(&self) -> Version;
    /// Модельные версии, которые плагин обслуживает.
    fn model_versions(&self) -> &'static [ModelVersion];
    /// Способности конкретной модельной версии.
    fn capabilities(&self, v: &ModelVersion) -> Capabilities;
    /// Внятный отказ для версии, которая есть в плане, но ещё не реализована
    /// (вместо «missing tensor» из недр загрузчика).
    fn planned_error(&self, v: &ModelVersion) -> Option<BackendError> {
        if v.implemented {
            return None;
        }
        Some(crate::arch::unsupported_arch(v.arch))
    }
}

/// Qwen: dense `v3`, гибрид `v3.5` (GDN; включая Qwen3.8-27B) и MoE
/// `v4exp` (Qwen3.8-Flash-Next — в плане, волна A5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenVersion {
    V3,
    V35,
    V4Exp,
}

impl QwenVersion {
    pub const ALL: [Self; 3] = [Self::V3, Self::V35, Self::V4Exp];

    pub fn name(self) -> &'static str {
        match self {
            Self::V3 => "v3",
            Self::V35 => "v3.5",
            Self::V4Exp => "v4exp",
        }
    }

    pub fn arch(self) -> &'static str {
        match self {
            Self::V3 => "qwen3",
            Self::V35 => "qwen35",
            Self::V4Exp => "qwen4exp",
        }
    }

    pub const fn model_version(self) -> ModelVersion {
        match self {
            Self::V3 => ModelVersion::new("v3", "qwen3", true),
            Self::V35 => ModelVersion::new("v3.5", "qwen35", true),
            Self::V4Exp => ModelVersion::planned(
                "v4exp",
                "qwen4exp",
                ".zcode/plans/plan-archs-and-tiered-inference.md (A5)",
            ),
        }
    }
}

impl fmt::Display for QwenVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Версия qwen-плагина по GGUF-архитектуре (`qwen3` → V3 и т.д.).
pub fn qwen_version_for_arch(arch: &str) -> Option<QwenVersion> {
    QwenVersion::ALL.into_iter().find(|v| v.arch() == arch)
}

/// Плагин движка Qwen. Один плагин, три модельные версии; способности
/// каждой объявлены явно, чтобы сервис мог маршрутизировать до загрузки.
pub struct QwenPlugin {
    pub engine_version: Version,
}

impl Default for QwenPlugin {
    fn default() -> Self {
        Self { engine_version: Version::new(1, 0, 0) }
    }
}

const QWEN_V3_CAPS: Capabilities = Capabilities {
    family: Family::Dense,
    implemented: true,
    kv_formats: &[KvFormat::F32, KvFormat::F16, KvFormat::Q8_0],
    speculative: &[SpecBackend::Off, SpecBackend::DFlash, SpecBackend::NGramLookup],
    offload: Offload::NONE,
    batch_slots: 4,
    max_context: 131_072,
    note: "Qwen3 dense: GQA, QK-norm, RoPE; 0.6–8B, vision на qwen3vl",
};

const QWEN_V35_CAPS: Capabilities = Capabilities {
    family: Family::Hybrid,
    implemented: true,
    kv_formats: &[KvFormat::F32, KvFormat::F16, KvFormat::Q8_0],
    speculative: &[
        SpecBackend::Off,
        SpecBackend::DFlash,
        SpecBackend::Mtp,
        SpecBackend::NGramLookup,
    ],
    offload: Offload::NONE,
    batch_slots: 4,
    max_context: 262_144,
    note: "Qwen3.5/3.8 hybrid (GDN-миксер + внимание): включает Qwen3.8-27B",
};

const QWEN_V4EXP_CAPS: Capabilities = Capabilities {
    family: Family::Moe,
    implemented: false,
    kv_formats: &[],
    speculative: &[SpecBackend::Off],
    offload: Offload { experts: true, kv: false, prefix_cache: false },
    batch_slots: 2,
    max_context: 262_144,
    note: "Qwen3.8-Flash-Next: hyper-connections, MoE 512/top-10 + shared, \
           QSA-индексер, PLE n-граммы; конфиг/схема — qwen4exp.rs, forward — A5",
};

impl EnginePlugin for QwenPlugin {
    fn id(&self) -> &str {
        "qwen"
    }

    fn kind(&self) -> PluginKind {
        PluginKind::Native
    }

    fn version(&self) -> Version {
        self.engine_version
    }

    fn model_versions(&self) -> &'static [ModelVersion] {
        const VERSIONS: [ModelVersion; 3] = [
            QwenVersion::V3.model_version(),
            QwenVersion::V35.model_version(),
            QwenVersion::V4Exp.model_version(),
        ];
        &VERSIONS
    }

    fn capabilities(&self, v: &ModelVersion) -> Capabilities {
        match qwen_version_for_arch(v.arch) {
            Some(QwenVersion::V3) => QWEN_V3_CAPS,
            Some(QwenVersion::V35) => QWEN_V35_CAPS,
            Some(QwenVersion::V4Exp) => QWEN_V4EXP_CAPS,
            None => Capabilities::from_arch(v.arch, &[], &[], Offload::NONE, 1, 0, "неизвестная версия"),
        }
    }
}

/// Внешний движок как плагин-адаптер: Infernix, Basalt, Strata, vLLM,
/// LM Studio, MLX-VLM, HyperQwen, … Регистрируется в рантайме сервисом
/// (или балансировщиком) по манифесту; способности — best-effort.
#[derive(Debug, Clone)]
pub struct ExternalEngine {
    pub id: String,
    /// Семейство движка: "infernix", "basalt", "strata", "vllm", "lmstudio",
    /// "mlx-vlm", "hyperqwen", "inferd", …
    pub engine: String,
    pub endpoint: String,
    pub engine_version: Option<Version>,
    pub models: Vec<ModelVersion>,
    pub capabilities: Capabilities,
}

impl ExternalEngine {
    pub fn new(
        id: impl Into<String>,
        engine: impl Into<String>,
        endpoint: impl Into<String>,
        models: Vec<ModelVersion>,
        capabilities: Capabilities,
    ) -> Self {
        Self {
            id: id.into(),
            engine: engine.into(),
            endpoint: endpoint.into(),
            engine_version: None,
            models,
            capabilities,
        }
    }
}

impl EnginePlugin for ExternalEngine {
    fn id(&self) -> &str {
        &self.id
    }

    fn kind(&self) -> PluginKind {
        PluginKind::External
    }

    fn version(&self) -> Version {
        self.engine_version.unwrap_or_default()
    }

    fn model_versions(&self) -> &'static [ModelVersion] {
        // Внешние движки перечисляют модели в рантайме — статического списка
        // у контракта нет. Сервис опрашивает `self.models` напрямую.
        &[]
    }

    fn capabilities(&self, v: &ModelVersion) -> Capabilities {
        if self.models.iter().any(|m| m.arch == v.arch || m.name == v.name) {
            self.capabilities
        } else {
            Capabilities::from_arch(v.arch, &[], &[], Offload::NONE, 0, 0, "модель не объявлена")
        }
    }
}

/// Реестр нативных плагинов (внешние добавляются сервисом в рантайме).
pub static QWEN: QwenPlugin = QwenPlugin { engine_version: Version::new(1, 0, 0) };

pub static NATIVE_REGISTRY: &[&dyn EnginePlugin] = &[&QWEN];

/// Проверка поддержки модели плагином: `Ok(plugin)` — есть реализованная
/// версия, `Err` — внятный отказ (включая «в плане») или «нет такого».
pub fn lookup_arch(arch: &str) -> Result<&'static dyn EnginePlugin, BackendError> {
    // qwen-версии покрывают все три архитектуры; другие плагины добавляются
    // сюда по мере появления.
    match qwen_version_for_arch(arch) {
        Some(QwenVersion::V3 | QwenVersion::V35) => Ok(&QWEN as &dyn EnginePlugin),
        Some(QwenVersion::V4Exp) => Err(crate::arch::unsupported_arch(arch)),
        None => {
            let known = NATIVE_REGISTRY
                .iter()
                .flat_map(|p| p.model_versions())
                .map(|m| m.arch)
                .collect::<Vec<_>>()
                .join(", ");
            Err(BackendError::UnsupportedFormat(format!(
                "архитектура {arch} не покрыта ни одним плагином; покрыты: {known}"
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_parse_and_order() {
        assert_eq!("1.0.0".parse::<Version>().unwrap(), Version::new(1, 0, 0));
        assert!("1.2".parse::<Version>().is_err());
        assert!("1.2.3.4".parse::<Version>().is_err());
        assert!("x.y.z".parse::<Version>().is_err());
        assert!(Version::new(0, 9, 0) < Version::new(0, 10, 0));
        assert_eq!(Version::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn qwen_plugin_serves_three_model_versions() {
        assert_eq!(QWEN.id(), "qwen");
        assert_eq!(QWEN.kind(), PluginKind::Native);
        let versions = QWEN.model_versions();
        assert_eq!(versions.len(), 3);
        assert_eq!(versions[0].name, "v3");
        assert_eq!(versions[1].name, "v3.5");
        assert_eq!(versions[2].name, "v4exp");
        assert!(versions[0].implemented);
        assert!(versions[1].implemented);
        assert!(!versions[2].implemented);
        assert!(versions[2].plan.is_some());
    }

    #[test]
    fn qwen_version_for_arch_maps_gguf_names() {
        assert_eq!(qwen_version_for_arch("qwen3"), Some(QwenVersion::V3));
        assert_eq!(qwen_version_for_arch("qwen35"), Some(QwenVersion::V35));
        assert_eq!(qwen_version_for_arch("qwen4exp"), Some(QwenVersion::V4Exp));
        assert_eq!(qwen_version_for_arch("llama"), None);
        assert_eq!(qwen_version_for_arch(""), None);
    }

    #[test]
    fn v35_capabilities_declare_27b_hybrid() {
        let caps = QWEN.capabilities(&QwenVersion::V35.model_version());
        assert!(caps.implemented);
        assert_eq!(caps.family, Family::Hybrid);
        assert_eq!(caps.max_context, 262_144);
        assert!(caps.kv_formats.contains(&KvFormat::Q8_0));
        assert!(!caps.offload.experts);
    }

    #[test]
    fn v4exp_is_planned_and_needs_expert_offload() {
        let v = QwenVersion::V4Exp.model_version();
        let caps = QWEN.capabilities(&v);
        assert!(!caps.implemented);
        assert_eq!(caps.family, Family::Moe);
        assert!(caps.offload.experts, "Flash-Next без offload экспертов не живёт");
        let err = QWEN.planned_error(&v).expect("планируемая версия обязана давать внятный отказ");
        let text = format!("{err}");
        assert!(text.contains("qwen4exp"), "отказ должен называть архитектуру: {text}");
    }

    #[test]
    fn external_engine_implements_the_same_contract() {
        let engine = ExternalEngine::new(
            "zeus",
            "infernix",
            "http://10.0.33.2:8095",
            vec![ModelVersion::new("v4exp", "qwen4exp", true)],
            QWEN_V4EXP_CAPS,
        );
        assert_eq!(engine.kind(), PluginKind::External);
        assert_eq!(engine.id(), "zeus");
        assert!(engine
            .capabilities(&ModelVersion::new("v4exp", "qwen4exp", true))
            .offload
            .experts);
    }
}
