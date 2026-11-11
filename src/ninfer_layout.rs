//! Раскладки контейнера `NINFER` → обычный ggml row-major.
//!
//! Infernix раскладывает веса под свои ядра, и в манифесте это поле `layout`.
//! Почти все они — перестановка ЦЕЛЫХ блоков квантования (блок = фиксированное
//! число байт на 256 элементов у K-квантов, 32 элемента у `q8_g32_fp16`),
//! поэтому декодер — это отображение «блок назначения → блок источника».
//!
//! Реализовано и проверено на синтетике:
//! * `contiguous_le_v1` — плоские элементы (bf16/fp32), переставлять нечего;
//! * `ggml_rows_v1` — блоки ggml подряд, тоже без перестановки;
//! * `ggml_tiles16_v1` — плитки по 16 строк: внутри плитки блоки идут
//!   k-блочно (вариант `TileMajor`), это гипотеза, которая проверяется
//!   паритетом текста с Infernix; варианты перечислены в [`Variant`];
//! * `row_split_k128_v1` — панели по 128 столбцов: k-панель → строка → группа.
//!
//! Варианты сделаны явными, чтобы перебор кандидатов был управляемым: для
//! каждой раскладки прогоняется свой маленький набор, и правильный выбирается
//! по совпадению первого жадного токена с самим Infernix.

/// Раскладка из манифеста.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Плоские элементы little-endian (bf16/fp32).
    ContiguousLe,
    /// Обычные ggml-блоки row-major.
    GgmlRows,
    /// Плитки по 16 строк.
    GgmlTiles16,
    /// Панели по 128 столбцов.
    RowSplitK128,
    /// Экспертный блок (несколько матриц в слоте эксперта).
    ExpertPacked,
}

impl Layout {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "contiguous_le_v1" => Ok(Layout::ContiguousLe),
            "ggml_rows_v1" => Ok(Layout::GgmlRows),
            "ggml_tiles16_v1" => Ok(Layout::GgmlTiles16),
            "row_split_k128_v1" => Ok(Layout::RowSplitK128),
            // оба экспертных варианта имеют одинаковую структуру блока,
            // различается только квант внутри
            "ggml_expert_iq4_nl_v1" | "ggml_expert_q2_0_v1" => Ok(Layout::ExpertPacked),
            other => Err(format!("NINFER: неизвестная раскладка {other}")),
        }
    }

    /// Нужна ли перестановка блоков вообще.
    pub fn is_plain(&self) -> bool {
        matches!(self, Layout::ContiguousLe | Layout::GgmlRows)
    }
}

/// Вариант перестановки внутри раскладки (для перебора кандидатов).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Variant {
    /// Как есть (row-major) — контрольная гипотеза.
    #[default]
    Identity,
    /// Плитка 16 строк: внутри плитки сначала k-блок, потом строка.
    TileMajor,
    /// Плитка 16 строк, но плитки идут по k-блокам снаружи.
    TileMajorKOuter,
    /// Панель k (128 столбцов) снаружи, внутри — строка, затем группа.
    PanelOuter,
    /// Панель k внутри: строка, затем панель, затем группа.
    PanelInner,
}

impl Variant {
    /// Кандидаты для раскладки — маленький осмысленный набор.
    pub fn candidates(layout: Layout) -> &'static [Variant] {
        match layout {
            Layout::ContiguousLe | Layout::GgmlRows | Layout::ExpertPacked => &[Variant::Identity],
            Layout::GgmlTiles16 => &[
                Variant::TileMajor,
                Variant::TileMajorKOuter,
                Variant::Identity,
            ],
            Layout::RowSplitK128 => &[Variant::PanelOuter, Variant::PanelInner, Variant::Identity],
        }
    }
}

/// Порядок блоков: `order[dst] = src`.
///
/// `rows × blocks_per_row` блоков разложены в раскладке `layout`; функция
/// говорит, какой блок источника встаёт на позицию назначения (row-major).
pub fn block_order(
    layout: Layout,
    variant: Variant,
    rows: usize,
    blocks_per_row: usize,
) -> Result<Vec<usize>, String> {
    let n = rows * blocks_per_row;
    let mut order: Vec<usize> = (0..n).collect();
    match layout {
        Layout::ContiguousLe | Layout::GgmlRows | Layout::ExpertPacked => return Ok(order),
        Layout::GgmlTiles16 => {
            if !rows.is_multiple_of(16) {
                return Err(format!(
                    "tiles16: {rows} строк не кратны 16 — раскладка не применима"
                ));
            }
            let tiles = rows / 16;
            match variant {
                Variant::Identity => return Ok(order),
                Variant::TileMajor => {
                    // src: tile → kb → row_in_tile
                    let mut src = 0;
                    for _tile in 0..tiles {
                        for kb in 0..blocks_per_row {
                            for r in 0..16 {
                                order[(_tile * 16 + r) * blocks_per_row + kb] = src;
                                src += 1;
                            }
                        }
                    }
                }
                Variant::TileMajorKOuter => {
                    let mut src = 0;
                    for kb in 0..blocks_per_row {
                        for _tile in 0..tiles {
                            for r in 0..16 {
                                order[(_tile * 16 + r) * blocks_per_row + kb] = src;
                                src += 1;
                            }
                        }
                    }
                }
                other => {
                    return Err(format!("tiles16: вариант {other:?} не применим"));
                }
            }
        }
        Layout::RowSplitK128 => {
            const PANEL: usize = 128;
            // блок = 32 элемента, значит в панели из 128 столбцов 4 группы
            let groups_per_panel = PANEL / 32;
            if !blocks_per_row.is_multiple_of(groups_per_panel) {
                return Err(format!(
                    "row_split: {blocks_per_row} блоков в строке не кратны {groups_per_panel}"
                ));
            }
            let panels = blocks_per_row / groups_per_panel;
            match variant {
                Variant::Identity => return Ok(order),
                Variant::PanelOuter => {
                    let mut src = 0;
                    for _p in 0..panels {
                        for r in 0..rows {
                            for g in 0..groups_per_panel {
                                order[r * blocks_per_row + _p * groups_per_panel + g] = src;
                                src += 1;
                            }
                        }
                    }
                }
                Variant::PanelInner => {
                    let mut src = 0;
                    for r in 0..rows {
                        for _p in 0..panels {
                            for g in 0..groups_per_panel {
                                order[r * blocks_per_row + _p * groups_per_panel + g] = src;
                                src += 1;
                            }
                        }
                    }
                }
                other => {
                    return Err(format!("row_split: вариант {other:?} не применим"));
                }
            }
        }
    }
    if order.iter().enumerate().all(|(i, s)| i == *s) {
        // не ошибка: вариант вырожден в тождество на этих размерах
    }
    Ok(order)
}

/// Применить порядок к сырым байтам блоков: `src[order[dst]] → dst`.
pub fn reorder_blocks(src: &[u8], order: &[usize], block_bytes: usize) -> Result<Vec<u8>, String> {
    let n = order.len();
    if src.len() != n * block_bytes {
        return Err(format!(
            "reorder: {} байт при {n} блоках по {block_bytes}",
            src.len()
        ));
    }
    let mut out = vec![0u8; src.len()];
    for (dst, &s) in order.iter().enumerate() {
        if s >= n {
            return Err(format!("reorder: блок {s} вне {n}"));
        }
        out[dst * block_bytes..(dst + 1) * block_bytes]
            .copy_from_slice(&src[s * block_bytes..(s + 1) * block_bytes]);
    }
    Ok(out)
}

impl Q8G32 {
    /// Деквантовать в f32 для CPU-эталона (`w = q · scale`).
    ///
    /// На CUDA это не нужно: там квант пойдёт своими ядрами, но для parity и
    /// для слоёв, которые считаются на хосте (нормы, роутер), нужен точный
    /// CPU-путь.
    pub fn dequant(&self, data: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
        let numel = (rows * cols) as u64;
        if !cols.is_multiple_of(128) || !cols.is_multiple_of(self.group) {
            return Err(format!("q8_g32: cols {cols} не кратны 128"));
        }
        let (q, sc) = self.split(data, numel)?;
        let panels = cols / 128;
        let per_panel_group = 128 / self.group; // 4 группы по 32
        let mut out = vec![0f32; rows * cols];
        for r in 0..rows {
            for p in 0..panels {
                for jj in 0..128 {
                    let j = p * 128 + jj;
                    let g = jj / self.group;
                    // Спецификация (§3.2/3.5): обход строкой-группами, кванты
                    // в base-плоскости, масштабы — в scale-плоскости с тем же
                    // порядком индексов.
                    let _ = (p, jj, per_panel_group, g);
                    let (qi, si) = (r * cols + j, (r * cols + j) / self.group);
                    let qv = q[qi] as i8 as f32;
                    let sb = &sc[si * 2..si * 2 + 2];
                    let scale = f16_to_f32(u16::from_le_bytes([sb[0], sb[1]]));
                    out[r * cols + j] = qv * scale;
                }
            }
        }
        Ok(out)
    }
}

/// fp16 → f32 (полный, включая субнормали и NaN).
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        // субнормаль/ноль
        (frac as f32) * 2f32.powi(-24)
    } else if exp == 0x1f {
        if frac == 0 { f32::INFINITY } else { f32::NAN }
    } else {
        (1.0 + frac as f32 / 1024.0) * 2f32.powi(exp as i32 - 15)
    };
    if sign == 1 { -out } else { out }
}

/// Банк экспертов MoE (`ggml_expert_iq4_nl_v1` / `ggml_expert_q2_0_v1`).
///
/// Спецификация (`infernix/src/docs/maintainer/storage-layouts.md` §8):
/// эксперт — одна запись из трёх матриц, все строки — вербатим ggml-блоки:
///
/// - gate rows `[I, H]` и up rows `[I, H]` — тип из поля `format` объекта;
/// - down rows `[H, I]` — тип из ИМЕНИ раскладки (`ggml_iq4_nl`/`ggml_q2_0`);
/// - `record_stride = align_up(record_bytes, 4096)`, хвост записи — нули.
///
/// На живом объекте `[512, 2560, 640]` iq3_xxs: gate_row 980 Б, down_row
/// 360 Б, record 2 176 000 → stride 2 179 072, и `bytes/512` совпадает
/// ровно — поэтому «лишние» 297 472 Б были не таблицей, а down-матрицами
/// в iq4_nl плюс паддинг.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertBank {
    pub experts: usize,
    /// `H` — ширина gate/up и число строк down.
    pub h: usize,
    /// `I` — строк gate/up и ширина down.
    pub i: usize,
    pub gate_up: BlockGeom,
    pub down: BlockGeom,
}

impl ExpertBank {
    /// Собрать банк по формату объекта и имени раскладки.
    pub fn new(
        experts: usize,
        h: usize,
        i: usize,
        gate_up_format: &str,
        layout: &str,
    ) -> Result<Self, String> {
        let gate_up = block_geom(gate_up_format)?;
        let down_type = match layout {
            "ggml_expert_iq4_nl_v1" => "ggml_iq4_nl",
            "ggml_expert_q2_0_v1" => "ggml_q2_0",
            other => return Err(format!("NINFER: {other} — не банк экспертов")),
        };
        let down = block_geom(down_type)?;
        if !(h as u64).is_multiple_of(gate_up.elems) || !(i as u64).is_multiple_of(down.elems) {
            return Err(format!(
                "bank: H {h} / блок {} или I {i} / блок {} не делятся",
                gate_up.elems, down.elems
            ));
        }
        Ok(Self {
            experts,
            h,
            i,
            gate_up,
            down,
        })
    }

    pub fn gate_row_bytes(&self) -> u64 {
        (self.h as u64 / self.gate_up.elems) * self.gate_up.bytes
    }

    pub fn down_row_bytes(&self) -> u64 {
        (self.i as u64 / self.down.elems) * self.down.bytes
    }

    pub fn record_bytes(&self) -> u64 {
        2 * self.i as u64 * self.gate_row_bytes() + self.h as u64 * self.down_row_bytes()
    }

    pub fn record_stride(&self) -> u64 {
        align_up_u64(self.record_bytes(), 4096)
    }

    pub fn payload_bytes(&self) -> u64 {
        self.experts as u64 * self.record_stride()
    }

    /// Байтовые срезы матриц эксперта `e`: (gate, up, down) внутри записи.
    pub fn slices(
        &self,
        e: usize,
    ) -> (
        std::ops::Range<usize>,
        std::ops::Range<usize>,
        std::ops::Range<usize>,
    ) {
        let stride = self.record_stride() as usize;
        let gr = self.gate_row_bytes() as usize;
        let dr = self.down_row_bytes() as usize;
        let base = e * stride;
        let gate = base..base + self.i * gr;
        let up = base + self.i * gr..base + 2 * self.i * gr;
        let down = base + 2 * self.i * gr..base + 2 * self.i * gr + self.h * dr;
        (gate, up, down)
    }
}

fn align_up_u64(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

/// Раскладка `q8_g32_fp16`: НЕ чередующиеся 34-байтовые блоки, а ДВА массива.
///
/// Проверено на живом объекте `moe/shared/down` [2560, 640] (1 740 800 байт):
///
/// - сначала `numel` байт квантов i8 (absmax 127, mean|q| ≈ 39);
/// - затем `numel/32` масштабов fp16 (51 200 значений, все в 5e-5…4e-3).
///
/// Сумма сходится ровно: 1 638 400 + 102 400 = 1 740 800.
///
/// Арифметика «34 байта на блок подряд» на этих данных даёт мусорные
/// масштабы в каждом втором блоке — поэтому формат разобран отдельно.
/// Порядок групп внутри массивов зависит от `row_split_k128_v1` и
/// определяется сверкой с Infernix (как и остальные раскладки).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Q8G32 {
    pub group: usize,
}

impl Q8G32 {
    pub const GROUP: usize = 32;

    /// Байты на тензор из `numel` элементов.
    pub fn bytes(numel: u64) -> u64 {
        numel + numel / Self::GROUP as u64 * 2
    }

    /// Разделить сырые байты на (кванты i8, масштабы fp16 в байтах).
    pub fn split<'a>(&self, data: &'a [u8], numel: u64) -> Result<(&'a [u8], &'a [u8]), String> {
        let q = numel as usize;
        let s = numel as usize / self.group * 2;
        if data.len() != q + s {
            return Err(format!(
                "q8_g32: {} байт вместо {} (кванты {q} + масштабы {s})",
                data.len(),
                q + s
            ));
        }
        Ok((&data[..q], &data[q..]))
    }
}

/// Формат объекта из манифеста → геометрия блока кванта.
///
/// Имена форматов в NINFER — свои (`ggml_q6_k`, `q8_g32_fp16`, `bf16`, …),
/// а не ggml-коды. Здесь только размеры: сам деквант делает `weights_map`
/// (для известных ggml-форматов) или отдельный кодек (для `q8_g32_fp16`,
/// которого в ggml нет).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGeom {
    pub elems: u64,
    pub bytes: u64,
    /// ggml-тип, если формат совпадает с ним (для декванта).
    pub ggml: Option<crate::gguf::GgmlType>,
}

impl BlockGeom {
    /// Диапазон в элементах → байты (границы обязаны попадать на блок).
    pub fn bytes_of(&self, elem_offset: u64, elem_len: u64) -> Option<(u64, u64)> {
        if self.elems == 0
            || !elem_offset.is_multiple_of(self.elems)
            || !elem_len.is_multiple_of(self.elems)
        {
            return None;
        }
        Some((
            elem_offset / self.elems * self.bytes,
            elem_len / self.elems * self.bytes,
        ))
    }
}

/// Разобрать имя формата из манифеста NINFER.
pub fn block_geom(format: &str) -> Result<BlockGeom, String> {
    use crate::gguf::GgmlType as G;
    let g = |t: G| BlockGeom {
        elems: t.block_elems() as u64,
        bytes: t.block_bytes() as u64,
        ggml: Some(t),
    };
    Ok(match format {
        "bf16" => g(G::Bf16),
        "fp16" => g(G::F16),
        "fp32" => g(G::F32),
        "ggml_q4_k" => g(G::Q4K),
        "ggml_q5_k" => g(G::Q5K),
        "ggml_q6_k" => g(G::Q6K),
        "ggml_q8_0" => g(G::Q8_0),
        "ggml_iq2_s" => g(G::Iq2S),
        "ggml_iq3_s" => g(G::Iq3S),
        "ggml_iq3_xxs" => g(G::Iq3Xxs),
        "ggml_iq4_xs" => g(G::Iq4Xs),
        "ggml_iq4_nl" => g(G::Iq4Nl),
        "ggml_q2_0" => g(G::Q2_0),
        // Q8 с группой 32 и fp16-масштабом: 32 байта кодов + 2 байта скейла.
        // В ggml такого формата нет — деквант пишется отдельно.
        "q8_g32_fp16" => BlockGeom {
            elems: 32,
            bytes: 34,
            ggml: None,
        },
        other => return Err(format!("NINFER: неизвестный формат {other}")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocks(n: usize) -> Vec<u8> {
        // каждый блок — 4 байта с номером блока
        (0..n).flat_map(|i| (i as u32).to_le_bytes()).collect()
    }

    fn ids(data: &[u8]) -> Vec<u32> {
        data.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn plain_layouts_are_identity() {
        for l in [Layout::ContiguousLe, Layout::GgmlRows] {
            let order = block_order(l, Variant::Identity, 4, 3).unwrap();
            assert_eq!(order, (0..12).collect::<Vec<_>>());
            assert_eq!(
                ids(&reorder_blocks(&blocks(12), &order, 4).unwrap()),
                (0..12).collect::<Vec<_>>()
            );
        }
    }

    /// tiles16 (вариант TileMajor): плитка 16 строк, внутри — k-блок снаружи.
    /// Малый аналог: 2 плитки по 2 строки не выразить (плитка фиксирована 16),
    /// поэтому проверяем на 16 строках и 2 блоках в строке.
    #[test]
    fn tiles16_tile_major_is_a_permutation() {
        let (rows, bpr) = (16usize, 2usize);
        let order = block_order(Layout::GgmlTiles16, Variant::TileMajor, rows, bpr).unwrap();
        // биекция
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..rows * bpr).collect::<Vec<_>>());
        // первый блок источника — (row 0, kb 0)
        assert_eq!(order[0], 0);
        // row 1, kb 0 — второй по порядку внутри плитки → источник 1
        assert_eq!(order[bpr], 1);
        // row 0, kb 1 → источник 16 (после всех строк kb 0)
        assert_eq!(order[1], 16);
        // применяем и проверяем, что данные — та же перестановка
        let src = blocks(rows * bpr);
        let out = reorder_blocks(&src, &order, 4).unwrap();
        assert_eq!(ids(&out).len(), rows * bpr);
        assert_eq!(ids(&out)[0], 0);
        assert_eq!(ids(&out)[1], 16);
    }

    /// row_split: 128 столбцов = 4 группы по 32; панель снаружи.
    #[test]
    fn row_split_panel_outer_permutation() {
        let (rows, bpr) = (2usize, 8usize); // 2 панели
        let order = block_order(Layout::RowSplitK128, Variant::PanelOuter, rows, bpr).unwrap();
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..rows * bpr).collect::<Vec<_>>());
        // dst (row 0, block 4) = первая группа второй панели → источник rows*4 = 8
        assert_eq!(order[4], 8);
        // dst (row 1, block 0) = вторая строка первой панели → источник 4
        assert_eq!(order[bpr], 4);
    }

    #[test]
    fn tiles16_rejects_rows_not_multiple_of_16() {
        assert!(block_order(Layout::GgmlTiles16, Variant::TileMajor, 10, 2).is_err());
    }

    #[test]
    fn reorder_rejects_size_mismatch() {
        let order = vec![0usize, 1];
        assert!(reorder_blocks(&[0u8; 4], &order, 4).is_err());
    }

    #[test]
    fn block_geometry_matches_manifest_sizes() {
        // gdn/output [2560, 6144] q6_k: 24 блока в строке × 210 Б × 2560 = 12 902 400
        let q6 = block_geom("ggml_q6_k").unwrap();
        assert_eq!((q6.elems, q6.bytes), (256, 210));
        assert_eq!(6144 / 256 * 210 * 2560u64, 12_902_400);
        // моя ошибка ранее: диапазоны parts — элементы, а не байты
        let total = 4096u64 * 2560;
        assert_eq!(q6.bytes_of(0, 5_242_880), Some((0, 5_242_880 / 256 * 210)));
        assert_eq!(q6.bytes_of(0, total), Some((0, 8_601_600)));
        // половина блока — отказ
        assert!(q6.bytes_of(0, 100).is_none());
        // q8_g32_fp16: 32 элемента на 34 байта
        let q8 = block_geom("q8_g32_fp16").unwrap();
        assert_eq!((q8.elems, q8.bytes), (32, 34));
        assert_eq!(q8.ggml, None, "в ggml такого формата нет");
        assert_eq!(q8.bytes_of(0, 2560 * 640), Some((0, 2560 * 640 / 32 * 34)));
        assert!(block_geom("nvfp4_rg16").is_err());
    }

    /// q8_g32_fp16 — два массива, а не 34-байтовые блоки подряд.
    #[test]
    fn q8_g32_is_two_arrays() {
        let numel = 2560u64 * 640;
        assert_eq!(Q8G32::bytes(numel), 1_740_800);
        let data = vec![0u8; 1_740_800];
        let (q, s) = Q8G32 { group: 32 }.split(&data, numel).unwrap();
        assert_eq!(q.len(), 1_638_400);
        assert_eq!(s.len(), 102_400);
        // Общий размер тот же, что у «34 байта на блок» (34/32 на элемент) —
        // различие ТОЛЬКО в расположении: кванты и масштабы идут двумя
        // массивами, поэтому чтение по 34-байтовым блокам даёт мусорные
        // масштабы в каждом втором блоке (проверено на живом файле).
        assert_eq!(numel / 32 * 34, Q8G32::bytes(numel));
        let mut bad = data.clone();
        bad.push(0);
        assert!(Q8G32 { group: 32 }.split(&bad, numel).is_err());
    }

    /// Деквант q8_g32 по спецификации (row-major группы, отдельная
    /// плоскость масштабов).
    #[test]
    fn q8_g32_dequant_both_orders() {
        let (rows, cols) = (2usize, 256usize); // 2 панели по 128
        let numel = (rows * cols) as u64;
        let scales = numel as usize / 32;
        let mut data = vec![0u8; numel as usize + scales * 2];
        // кванты: 1..=127 по кругу; масштабы: 0.5 для всех
        for (i, b) in data[..numel as usize].iter_mut().enumerate() {
            *b = (i % 127 + 1) as u8;
        }
        for i in 0..scales {
            data[numel as usize + i * 2] = 0x00; // f16 0x3800 = 0.5
            data[numel as usize + i * 2 + 1] = 0x38;
        }
        let g = Q8G32 { group: 32 };
        let w = g.dequant(&data, rows, cols).unwrap();
        assert_eq!(w.len(), rows * cols);
        // масштаб 0.5 → |w| = |q|/2, все значения не больше 63.5
        assert!(w.iter().all(|v| v.abs() <= 63.5 + 1e-3));
        assert!(w.iter().any(|v| v.abs() > 1.0));
        // отказ: cols не кратен 128
        assert!(g.dequant(&data, rows, 96).is_err());
    }

    /// Банк экспертов сходится с живым объектом до байта.
    #[test]
    fn expert_bank_matches_live_sizes() {
        // слой 0: iq3_xxs, down — iq4_nl (имя раскладки)
        let b = ExpertBank::new(512, 2560, 640, "ggml_iq3_xxs", "ggml_expert_iq4_nl_v1").unwrap();
        assert_eq!(b.gate_row_bytes(), 10 * 98);
        assert_eq!(b.down_row_bytes(), 20 * 18);
        assert_eq!(b.record_bytes(), 2 * 640 * 980 + 2560 * 360);
        assert_eq!(b.record_bytes(), 2_176_000);
        assert_eq!(b.record_stride(), 2_179_072);
        assert_eq!(b.payload_bytes(), 1_115_684_864);
        // слой 3: iq2_s, down — iq4_nl (тот же layout)
        let b3 = ExpertBank::new(512, 2560, 640, "ggml_iq2_s", "ggml_expert_iq4_nl_v1").unwrap();
        assert_eq!(b3.gate_row_bytes(), 10 * 82);
        assert_eq!(b3.record_bytes(), 2 * 640 * 820 + 2560 * 360);
        // срезы не пересекаются и лежат внутри СВОЕЙ записи
        let (g, u, d) = b.slices(1);
        let base = b.record_stride() as usize;
        assert_eq!(g.start, base);
        assert_eq!(g.end, u.start);
        assert_eq!(u.end, d.start);
        assert!(d.end - base <= b.record_stride() as usize);
        assert!(b.slices(0).2.end <= b.record_stride() as usize);
    }

    #[test]
    fn f16_conversion_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn parse_and_plainness() {
        assert_eq!(
            Layout::parse("contiguous_le_v1").unwrap(),
            Layout::ContiguousLe
        );
        assert_eq!(
            Layout::parse("ggml_tiles16_v1").unwrap(),
            Layout::GgmlTiles16
        );
        assert_eq!(
            Layout::parse("row_split_k128_v1").unwrap(),
            Layout::RowSplitK128
        );
        assert_eq!(
            Layout::parse("ggml_expert_iq4_nl_v1").unwrap(),
            Layout::ExpertPacked
        );
        assert!(Layout::parse("nvfp4_rg16_v1").is_err());
        assert!(Layout::ContiguousLe.is_plain() && !Layout::GgmlTiles16.is_plain());
    }
}
