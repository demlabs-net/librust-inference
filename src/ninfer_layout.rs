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
