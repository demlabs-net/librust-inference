//! Загрузчик весов Flash-Next из контейнера NINFER.
//!
//! Задача загрузчика — превратить манифест Infernix в плоские тензоры в
//! обычном ggml-порядке, которые ждёт forward:
//!
//! 1. **Собрать объект**: объект может пересекать тома контейнера, поэтому
//!    читается через `slice_ranges` и склеивается.
//! 2. **Распутать раскладку**: `ggml_tiles16_v1` / `row_split_k128_v1` — это
//!    перестановка ЦЕЛЫХ блоков кванта ([`crate::ninfer_layout`]); вариант
//!    выбирается прогоном против Infernix (первый жадный токен), а не на глаз.
//!    Перестановка применяется к ОБЪЕКТУ, потому что раскладка описывает весь
//!    объект; логические части режутся уже после неё.
//! 3. **Вырезать логический тензор**: `parts[].range` — диапазон в ЭЛЕМЕНТАХ
//!    объекта (проверено на живом файле), границы кратны строке и блоку
//!    кванта; результат — непрерывные строки `[rows, cols]`.
//!
//! Всё остаётся в нативном кванте: деквант делает тот, кто грузит тензор на
//! бэкенд (`from_quantized`/`dequant_gguf`). Так 51 ГиБ весов не раздуваются
//! в f32 при открытии модели.

use crate::error::BackendError;
use crate::ninfer::{Ninfer, Object};
use crate::ninfer_layout::{
    BlockGeom, ExpertBank, Layout, Variant, block_geom, block_order, reorder_blocks,
};

/// Логический тензор в ggml-порядке (нативный квант, без декванта).
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    /// `[rows, cols]` в нашем порядке (GGUF `ne0` — это `cols`).
    pub rows: usize,
    pub cols: usize,
    /// Исходный формат манифеста (`ggml_q6_k`, `bf16`, …).
    pub format: String,
    pub geom: BlockGeom,
    /// Байты в ggml row-major (каждая строка — целое число блоков кванта).
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn bytes(&self) -> usize {
        self.data.len()
    }
}

/// Матрицы одного эксперта в нативных квантах.
#[derive(Debug, Clone)]
pub struct ExpertMatrices {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
    pub bank: ExpertBank,
}

/// Открытые веса: контейнер + выбранные варианты раскладок.
pub struct Weights {
    container: Ninfer,
    /// Вариант для `ggml_tiles16_v1`.
    tiles: Variant,
    /// Вариант для `row_split_k128_v1`.
    split: Variant,
}

impl std::fmt::Debug for Weights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Weights")
            .field("artifact", &self.container.name)
            .field("tiles", &self.tiles)
            .field("split", &self.split)
            .finish()
    }
}

impl Weights {
    /// Открыть контейнер. Варианты раскладок задаются явно: перебор кандидатов
    /// ведёт вызывающий (сверка первого токена с Infernix).
    pub fn open(
        path: &std::path::Path,
        tiles: Variant,
        split: Variant,
    ) -> Result<Self, BackendError> {
        let container = Ninfer::open(path).map_err(BackendError::UnsupportedFormat)?;
        Ok(Self {
            container,
            tiles,
            split,
        })
    }

    pub fn artifact(&self) -> &str {
        &self.container.name
    }

    /// Имена bindings с префиксом (диагностика/карта слоёв).
    pub fn names_with_prefix(&self, prefix: &str) -> Vec<String> {
        self.container
            .names_with_prefix(prefix)
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    fn variant_for(&self, layout: Layout) -> Variant {
        match layout {
            Layout::GgmlTiles16 => self.tiles,
            Layout::RowSplitK128 => self.split,
            _ => Variant::Identity,
        }
    }

    /// Прочитать логический тензор по bindings-имени в ggml row-major.
    ///
    /// Части (если их несколько, как у `gdn/query`+`gdn/key`) склеиваются —
    /// это ровно то, что ждёт fused-тензор (`attn_qkv` = q|k|v).
    pub fn tensor(&self, name: &str) -> Result<Tensor, BackendError> {
        let parts = self
            .container
            .parts(name)
            .map_err(BackendError::UnsupportedFormat)?;
        let mut out: Option<Tensor> = None;

        for (obj, elem_off, elem_len) in parts {
            let geom = block_geom(&obj.format).map_err(BackendError::UnsupportedFormat)?;
            let layout = Layout::parse(&obj.layout).map_err(BackendError::UnsupportedFormat)?;
            let cols = obj.shape.first().copied().unwrap_or(0) as usize;
            if cols == 0 || obj.shape.len() != 2 {
                return Err(BackendError::UnsupportedFormat(format!(
                    "NINFER {name}: ожидался 2-Д объект, у {} {:?}",
                    obj.id, obj.shape
                )));
            }
            let rows_total = obj.numel() as usize / cols;
            let blocks_per_row = cols / geom.elems as usize;

            // 1. Объект целиком (может пересекать тома), 2. раскладка → row-major.
            let raw = self
                .container
                .read(obj)
                .map_err(BackendError::UnsupportedFormat)?;
            let order = block_order(layout, self.variant_for(layout), rows_total, blocks_per_row)
                .map_err(BackendError::UnsupportedFormat)?;
            let plain = if layout.is_plain() {
                raw
            } else {
                reorder_blocks(&raw, &order, geom.bytes as usize)
                    .map_err(BackendError::UnsupportedFormat)?
            };

            // 3. Часть = целые строки после перестановки.
            if !elem_off.is_multiple_of(cols as u64) || !elem_len.is_multiple_of(cols as u64) {
                return Err(BackendError::UnsupportedFormat(format!(
                    "NINFER {name}: диапазон [{elem_off}, {}] не кратен строке {cols}",
                    elem_off + elem_len
                )));
            }
            let rows = (elem_len / cols as u64) as usize;
            let (b0, blen) = geom.bytes_of(elem_off, elem_len).ok_or_else(|| {
                BackendError::UnsupportedFormat(format!(
                    "NINFER {name}: диапазон не кратен блоку {} элементов",
                    geom.elems
                ))
            })?;
            let data = plain[b0 as usize..(b0 + blen) as usize].to_vec();

            match &mut out {
                None => {
                    out = Some(Tensor {
                        name: name.to_string(),
                        rows,
                        cols,
                        format: obj.format.clone(),
                        geom,
                        data,
                    })
                }
                Some(acc) => {
                    // Несколько частей = конкатенация по строкам (q|k, alpha|beta).
                    if acc.cols != cols || acc.geom != geom {
                        return Err(BackendError::UnsupportedFormat(format!(
                            "NINFER {name}: части с разной геометрией ({}×{} против {cols}×?)",
                            acc.rows, acc.cols
                        )));
                    }
                    acc.rows += rows;
                    acc.data.extend_from_slice(&data);
                }
            }
        }
        out.ok_or_else(|| BackendError::UnsupportedFormat(format!("NINFER: пустой тензор {name}")))
    }

    /// Список объектов-экспертов (3-Д) — банков MoE по одному на слой.
    pub fn expert_objects(&self) -> Vec<&Object> {
        self.container
            .objects()
            .iter()
            .filter(|o| o.is_tensor() && o.shape.len() == 3)
            .collect()
    }

    /// Банк экспертов слоя и сам объект: геометрия — по спецификации
    /// (`ggml_expert_iq4_nl_v1`: gate/up — тип из `format`, down — из имени
    /// раскладки, страйд `align_up(record, 4096)`).
    pub fn expert_bank(&self, layer: usize) -> Result<ExpertBank, BackendError> {
        let name = format!("text/layers/{layer}/moe/experts");
        let (obj, _, _) = self
            .container
            .parts(&name)
            .map_err(BackendError::UnsupportedFormat)?
            .into_iter()
            .next()
            .ok_or_else(|| BackendError::UnsupportedFormat(format!("NINFER: пустой {name}")))?;
        // shape в GGUF-порядке: [I, H, E] (ne0 — ширина down? нет: см. ниже)
        if obj.shape.len() != 3 {
            return Err(BackendError::UnsupportedFormat(format!(
                "{name}: ожидался 3-Д банк, у {} {:?}",
                obj.id, obj.shape
            )));
        }
        // Манифест: [512, 2560, 640] = [E, H, I] (см. storage-layouts §8),
        // поэтому H = shape[1], I = shape[2], экспертов = shape[0].
        let experts = obj.shape[0] as usize;
        let h = obj.shape[1] as usize;
        let i = obj.shape[2] as usize;
        let bank = ExpertBank::new(experts, h, i, &obj.format, &obj.layout)
            .map_err(BackendError::UnsupportedFormat)?;
        if bank.payload_bytes() != obj.bytes {
            return Err(BackendError::UnsupportedFormat(format!(
                "{name}: по спецификации {} байт, в объекте {}",
                bank.payload_bytes(),
                obj.bytes
            )));
        }
        Ok(bank)
    }

    /// Одна матрица эксперта `e`: gate (`[I,H]`), up (`[I,H]`), down (`[H,I]`).
    /// Читается ТОЛЬКО запись эксперта — 1.1 ГБ объекта целиком не нужны.
    pub fn expert(&self, layer: usize, e: usize) -> Result<ExpertMatrices, BackendError> {
        let bank = self.expert_bank(layer)?;
        if e >= bank.experts {
            return Err(BackendError::UnsupportedFormat(format!(
                "эксперт {e} вне банка ({} шт)",
                bank.experts
            )));
        }
        let name = format!("text/layers/{layer}/moe/experts");
        let (obj, _, _) = self
            .container
            .parts(&name)
            .map_err(BackendError::UnsupportedFormat)?
            .into_iter()
            .next()
            .expect("parts непустой (проверено в expert_bank)");
        let stride = bank.record_stride();
        let record = Object {
            offset: obj.offset + e as u64 * stride,
            bytes: bank.record_bytes(),
            ..obj.clone()
        };
        let raw = self
            .container
            .read(&record)
            .map_err(BackendError::UnsupportedFormat)?;
        let (g, u, d) = bank.slices(0); // срезы внутри записи
        Ok(ExpertMatrices {
            gate: raw[g].to_vec(),
            up: raw[u].to_vec(),
            down: raw[d].to_vec(),
            bank,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ninfer::{MAGIC, MAGIC_PART, PAYLOAD_ALIGN, VERSION};

    fn write_container(dir: &std::path::Path) -> std::path::PathBuf {
        // Объект bf16 shape [8, 4] в GGUF-порядке (ne0=8 — это cols):
        // 32 элемента = 4 строки по 8. bindings:
        // q = строки 0..4, k = строки 4..8, цельный = весь объект.
        let json = serde_json::json!({
            "metadata": { "name": "test" },
            "objects": [
                { "id": "weight/0", "kind": "tensor", "shape": [8, 4], "format": "bf16",
                  "layout": "contiguous_le_v1", "offset": 0, "bytes": 64 }
            ],
            "bindings": {
                "a/q": { "parts": [{ "object": "weight/0", "range": [0, 16] }] },
                "a/k": { "parts": [{ "object": "weight/0", "range": [16, 32] }] },
                "a/all": { "object": "weight/0" }
            },
            "files": [{ "path": null, "payload_bytes": 64 }]
        });
        let body = serde_json::to_vec(&json).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(MAGIC);
        file.push(VERSION);
        file.extend_from_slice(&(body.len() as u64).to_le_bytes());
        file.extend_from_slice(&[0u8; 16]);
        file.extend_from_slice(&body);
        while file.len() % PAYLOAD_ALIGN as usize != 0 {
            file.push(0);
        }
        // payload: блочные номера 0..31
        for i in 0u32..32 {
            file.extend_from_slice(&(i as u16).to_le_bytes());
        }
        let p = dir.join("t.ninfer");
        std::fs::write(&p, &file).unwrap();
        p
    }

    #[test]
    fn reads_parts_as_rows() {
        let dir = std::env::temp_dir().join(format!("q4e-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_container(&dir);
        let w = Weights::open(&path, Variant::Identity, Variant::Identity).unwrap();
        assert_eq!(w.artifact(), "test");

        let all = w.tensor("a/all").unwrap();
        assert_eq!((all.rows, all.cols), (4, 8));
        assert_eq!(all.bytes(), 64);

        // q = первые 2 строки (16 элементов = 2×8)
        let q = w.tensor("a/q").unwrap();
        assert_eq!((q.rows, q.cols), (2, 8));
        assert_eq!(q.data.len(), 32);
        // bf16: значения 0,1,2,...
        let first = u16::from_le_bytes([q.data[0], q.data[1]]);
        assert_eq!(first, 0);
        let k = w.tensor("a/k").unwrap();
        assert_eq!((k.rows, k.cols), (2, 8));
        assert_eq!(u16::from_le_bytes([k.data[0], k.data[1]]), 16);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// tiles16: перестановка применяется к ОБЪЕКТУ, и только потом режутся
    /// части — иначе логический тензор собрался бы из чужих блоков.
    #[test]
    fn tiles16_permutes_whole_object_before_parts() {
        let dir = std::env::temp_dir().join(format!("q4e-t16-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // объект [16, 2] (GGUF ne0=2 → cols=2, rows=16), bf16, tiles16
        let json = serde_json::json!({
            "metadata": { "name": "t16" },
            "objects": [
                { "id": "w", "kind": "tensor", "shape": [2, 16], "format": "bf16",
                  "layout": "ggml_tiles16_v1", "offset": 0, "bytes": 64 }
            ],
            "bindings": {
                "x/top": { "parts": [{ "object": "w", "range": [0, 16] }] },
                "x/bot": { "parts": [{ "object": "w", "range": [16, 32] }] }
            },
            "files": [{ "path": null, "payload_bytes": 64 }]
        });
        let body = serde_json::to_vec(&json).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(MAGIC);
        file.push(VERSION);
        file.extend_from_slice(&(body.len() as u64).to_le_bytes());
        file.extend_from_slice(&[0u8; 16]);
        file.extend_from_slice(&body);
        while file.len() % PAYLOAD_ALIGN as usize != 0 {
            file.push(0);
        }
        for i in 0u32..32 {
            file.extend_from_slice(&(i as u16).to_le_bytes());
        }
        let path = dir.join("t16.ninfer");
        std::fs::write(&path, &file).unwrap();

        // Без перестановки верхние 8 строк = элементы 0..16.
        let w0 = Weights::open(&path, Variant::Identity, Variant::Identity).unwrap();
        let top0 = w0.tensor("x/top").unwrap();
        assert_eq!(u16::from_le_bytes([top0.data[0], top0.data[1]]), 0);

        // С TileMajor блоки идут «k-блок снаружи»: первая строка каждого блока
        // берётся из начала объекта, поэтому первый элемент верхних строк
        // остаётся 0, но вторая строка (элемент 2) — уже из другого места.
        let w1 = Weights::open(&path, Variant::TileMajor, Variant::Identity).unwrap();
        let top1 = w1.tensor("x/top").unwrap();
        assert_eq!((top1.rows, top1.cols), (8, 2));
        // раскладка: out[1] — второй k-блок первой строки (источник 16),
        // out[2] — первая строка второго k-блока? Нет: out[2] = (row 1, kb 0)
        // = источник 1; out[1] = (row 0, kb 1) = источник 16.
        assert_eq!(u16::from_le_bytes([top1.data[0], top1.data[1]]), 0);
        assert_eq!(u16::from_le_bytes([top1.data[2], top1.data[3]]), 16);
        assert_eq!(u16::from_le_bytes([top1.data[4], top1.data[5]]), 1);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Банк экспертов: запись читается по спецификации, срезы не пересекаются,
    /// xвост записи — паддинг до 4096.
    #[test]
    fn expert_record_slices_follow_spec() {
        let dir = std::env::temp_dir().join(format!("q4e-exp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // H=256 (1 блок gate/up), I=32 (1 группа down iq4_nl),
        // record = 2·32·98 + 256·18 = 10 880 → stride 12 288, E=2.
        let (e, h, i) = (2usize, 256usize, 32usize);
        let stride = 12288usize;
        let json = serde_json::json!({
            "metadata": { "name": "exp" },
            "objects": [
                { "id": "w", "kind": "tensor", "shape": [e, h, i], "format": "ggml_iq3_xxs",
                  "layout": "ggml_expert_iq4_nl_v1", "offset": 0, "bytes": stride * e }
            ],
            "bindings": { "text/layers/0/moe/experts": { "object": "w" } },
            "files": [{ "path": null, "payload_bytes": stride * e }]
        });
        let body = serde_json::to_vec(&json).unwrap();
        let mut file = Vec::new();
        file.extend_from_slice(MAGIC);
        file.push(VERSION);
        file.extend_from_slice(&(body.len() as u64).to_le_bytes());
        file.extend_from_slice(&[0u8; 16]);
        file.extend_from_slice(&body);
        while file.len() % PAYLOAD_ALIGN as usize != 0 {
            file.push(0);
        }
        // payload: экспертные записи залиты своим номером
        for ex in 0..e {
            file.extend(std::iter::repeat(ex as u8 + 1).take(stride));
        }
        let path = dir.join("e.ninfer");
        std::fs::write(&path, &file).unwrap();

        let w = Weights::open(&path, Variant::TileMajor, Variant::Identity).unwrap();
        let bank = w.expert_bank(0).unwrap();
        assert_eq!(bank.experts, e);
        assert_eq!(bank.record_bytes(), 10880);
        assert_eq!(bank.record_stride() as usize, stride);
        assert_eq!(bank.payload_bytes() as usize, stride * e);

        let e0 = w.expert(0, 0).unwrap();
        assert_eq!(e0.gate.len(), 32 * 98);
        assert_eq!(e0.up.len(), 32 * 98);
        assert_eq!(e0.down.len(), 256 * 18);
        assert!(e0.gate.iter().all(|b| *b == 1), "запись 0 залита единицами");
        let e1 = w.expert(0, 1).unwrap();
        assert!(e1.down.iter().all(|b| *b == 2), "запись 1 — двойками");
        assert!(w.expert(0, 2).is_err(), "эксперт вне банка");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_unknown_format() {
        let dir = std::env::temp_dir().join(format!("q4e-load-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = write_container(&dir);
        let w = Weights::open(&path, Variant::Identity, Variant::Identity).unwrap();
        assert!(w.tensor("a/none").is_err(), "нет такого bindings-имени");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
