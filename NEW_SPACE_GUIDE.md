# ColorBench — Yeni Uzay Geliştirme Rehberi

Bu doküman, ColorBench'in mevcut durumunu ve yeni bir renk uzayı denerken bilmen gereken her şeyi tek yerde tutar. **Yeni bir uzay üretmeye başlamadan önce burayı oku.**

> Son güncelleme: 2026-05-07. Snapshot temeli: OKLab + Helmgen v0.11.1 (Phase 11 final).

---

## 1. ColorBench iki ayrı pipeline'dan oluşur

| Pipeline | Amaç | Giriş noktası |
|---|---|---|
| **Generation pipeline** | Üretim uzayları (palette, gradient, gamut map, picker, end-user perceptual) | `python run.py oklab mynew` |
| **Measurement pipeline** | Ölçüm uzayları (ΔE STRESS, 3 dataset) | `python run.py metric --json params.json` |

İki pipeline aynı uzayı **farklı açılardan** test eder. Yeni uzay tasarlarken hangisinde rekabet ettiğini en başta belirle.

---

## 2. Generation pipeline detayı

### 2.1 Modüller ve fonksiyon sayısı

| Modül | `measure_*` fonksiyon | Konu |
|---|---:|---|
| `core/gpu_metrics.py` | 8 | Round-trip, achromatic, gradient, gamut, gamut_mapping, hue, special_gradients, stability |
| `core/gpu_metrics_perceptual.py` | 14 | Munsell, MacAdam, palette uniformity, tint/shade, dataviz, multistop, WCAG, harmony, photo gamut, animation, hue_agreement, shade hue, chroma preservation |
| `core/gpu_metrics_advanced.py` | 17 | CVD, animation, extremes, jacobian, contrast, hue leaf, 3color, double_rt, cross_gamut, quantization, channel_mono, banding, oog, hue_reversal, primary_disc, neg_lms, extreme_chroma |
| `core/gpu_metrics_independent.py` | 3 | Hung-Berns, Ebner-Fairchild, Pointer (3rd-party datasets) |
| `core/gpu_metrics_user_full.py` | **39** | Phase 9-11 end-user perceptual (real photo, palettes, skin, picker, CVD, display, state, print, HDR, glassmorphism, JND-aware) |
| **TOPLAM** | **81 fonksiyon** | |

### 2.2 Snapshot dökümü

- **90 top-level metric grubu** (her `measure_*` → 1+ grup)
- **1666 flat scalar metric** (alt-detaylar dahil, snapshot regression için dondurulmuş)
- Hepsi colour-science ground truth (`gpu_de.py` Sharma 2005 + colour-science bit-identical, 4.44e-16)
- Tam deterministik (her random `manual_seed` ile sabit)

### 2.3 Kategori taxonomy (`core/metric_categories.py`)

Her metric grubu 4 meta kategoriden birine eşlenir:

| Meta kategori | Anlamı | Karar değeri |
|---|---|---|
| `mathematical` | Round-trip, jacobian, condition number | Düşük; "uzay matematiksel olarak sağlam mı?" |
| `structural` | Gamut hacmi, achromatic axis, primary disc | Orta; uzayın kendi iç tutarlılığı |
| `perceptual_internal` | Munsell, MacAdam, hue uniformity (ölçek-içi) | Yüksek; klasik perceptual literatür |
| `perceptual_visible` | End-user görsel gerçekliği (palette, photo, picker, dark mode) | **En yüksek**; tasarımcı gerçekten görür mü? |

**Yeni uzay denerken ana hedefin `perceptual_visible` olmalı.** Diğerleri matematiksel zafer; bu sonuncusu pazarda fark yaratır.

---

## 3. Measurement pipeline detayı

### 3.1 Modül ve metodoloji

| Bileşen | Detay |
|---|---|
| Modül | `core/metric_eval.py` (tek dosya) |
| Output metric | **STRESS** (CIE 224:2017 standardı) |
| Pre-processing | Bradford CAT (per-pair, source illuminant → D65) |
| ΔE baseline | 8 fonksiyon — CIE76, CIEDE2000, CIE94, OKLab, CAM16-UCS, CIECAM02-UCS, JzAzBz, DIN99 |

### 3.2 Datasets

| Dataset | Pair sayısı | Not |
|---|---:|---|
| COMBVD | 3,813 | Birleşik visual difference dataset |
| MacAdam 1974 | 128 | Klasik thresholds |
| Human Feedback | 3,552 | Self-collected (kültürel bias riski var) |
| **Toplam (3 dataset)** | **7,575** | |
| He 2022 (held-out) | 82 | 3D-printed; **eğitime girmez**, generalization göstergesi |

### 3.3 Mevcut leaderboard (MetricSpace v21)

| Dataset | v21 STRESS | CIEDE2000 STRESS | Kazanç |
|---|---:|---:|---:|
| COMBVD | **22.48** | 29.20 | %23 daha iyi |
| MacAdam | **19.51** | 22.13 | %12 daha iyi |
| HumFB | **23.26** | 62.54 | %63 daha iyi |
| He 2022 (held-out) | 35.9 | 32.6 | **CIEDE2000 daha iyi** |

⚠ He 2022 sonucu önemli: v21 COMBVD üzerinde **eğitildiği için** holdout'ta zayıflıyor. Yeni ölçüm uzayı dene­diğinde He 2022'yi mutlaka rapor et.

---

## 4. Pin testler & snapshot regression

### 4.1 7 pin test dosyası (32+ aktif test)

| Dosya | Test sayısı | Konu |
|---|---:|---|
| `tests/test_baseline_pin_colour.py` | 6 | Inline ΔE colour-science bit-identical |
| `tests/test_canonical_pin.py` | 6 | Canonical wrapper |
| `tests/test_gpu_de_pin.py` | 2 | 200 random + Sharma 2005 reference |
| `tests/test_literature_pin_colour.py` | 5 | 1 PASS, 4 documented drift |
| `tests/test_snapshot_regression.py` | 6 | Live OKLab vs snapshot, 1666 flat metric |
| `tests/test_spaces_pin_colour.py` | 7 | Production spaces forward/inverse |
| `tests/test_user_perceptual_pin.py` | 4 | 39 user_* smoke + determinism + categories + scalar |

### 4.2 Snapshot dosyaları

`tests/snapshots/` altında: OKLab + Helmgen v0.11.1 (2026-05-07_v6 final, Phase 11 fixed).

Yeni uzay eklediğinde:
1. `pytest colorbench/tests/` — 32+ pin test PASS olmalı
2. Snapshot regression: 1666 flat metric değişmediğini doğrular (mevcut uzaylar için)
3. Yeni uzayın kendi snapshot'ı oluşturulur (auto)

---

## 5. Adillik dürüst değerlendirme

### 5.1 Generation pipeline — %95 emin

✅ **Tam emin olduklarım:**
- ΔE temel matematik (gpu_de.py Sharma 2005 referans, colour-science bit-identical)
- Determinizm (her random explicit seed)
- Datasets reproducible (sabit JSON)
- Phase 11 fairness fix (test-space-coord → CIE Lab hedef, parametre-semantik fix)
- Snapshot regression korumalı

⚠ **%100 emin olmadıklarım:**
- **Lab-cartesian ailesi bias**: Tüm production uzayları Lab cartesian (OKLab, Helmgen, CIELab, IPT). LCh polar testleri özel ele alınmadı (real-world senaryolar dışında). Polar paradigma bir uzay düşük puan alabilir.
- **JND-relativity**: 1666 metricin ~%30'u sub-JND (matematik kazanım, görsel sıfır). `user_jnd_aware_summary` tek metric — daha sistematik flag yok.
- **API parity**: ColorBench Lab-linear interp her uzaya aynı uygulanır (adil), ama Helmgen native `hl.gradient()` arc-length avantajını yansıtmaz. Native API'lerin avantajları test dışı kalır.
- **Subjective study yok**: tüm metric otomatik. "Tasarımcı gerçekten görür mü?" cevap vermez.

### 5.2 Measurement pipeline — %90 emin

✅ **Emin olduklarım:**
- STRESS metodolojisi CIE 224:2017 standardı
- 8 baseline ΔE colour-science wrapper, bit-identical
- Bradford CAT per-pair correctly
- COMBVD/MacAdam/HumFB datasets stable

⚠ **%100 emin olmadıklarım (ama düzeltilmesi gerekmiyor):**
- **Dataset bias**: v21 COMBVD üzerinde eğitildi → COMBVD STRESS'i overfitted. He 2022 held-out'ta CIEDE2000 daha iyi (35.9 vs 32.6). **Aksiyon yok**: He holdout kuralı zaten var, paper §6.5 dürüstçe rapor ediyor.
- **HumFB self-collected**: 71 gözlemci, 3552 yargı → küçük örneklem, kültürel bias. **Aksiyon yok**: paper §6 dürüstçe rapor ediyor, leaderboard'da `*` ile işaretlenebilir ama metric'i değiştirmez.
- **CAT methodology**: Bradford vs CAT16 vs CAT02 vs Von Kries — **2026-05-07 test edildi, gürültü seviyesi**:

  | CAT | COMBVD CIEDE2000 STRESS |
  |---|---:|
  | Bradford | 29.1324 |
  | CAT16 | 29.1402 |
  | CAT02 | 29.1366 |
  | Von Kries | 29.1260 |

  Max spread: **0.014 STRESS**. v21'in 6.65 STRESS kazancının %0.12'si. **Aksiyon yok**: CAT seçimi sonucu anlamlı şekilde değiştirmiyor. Bradford yeterli.

---

## 6. Yeni uzay eklemek — adım adım

### 6.1 5 adımda çalışır

```python
# 1) colorbench/core/spaces.py'a yeni class ekle:
class MyNewSpace(ColorSpace):
    name = "MyNewSpace"
    def __init__(self, device):
        # parametreler, matrisler, transfer fonksiyonu
        ...
    def forward(self, xyz):  # XYZ → Lab-benzeri
        ...
    def inverse(self, lab):  # Lab-benzeri → XYZ
        ...

# 2) colorbench/run.py'a register:
elif s == "mynew":
    return MyNewSpace(device)

# 3) Çalıştır:
python run.py oklab mynew

# 4) Pin testleri çalıştır:
pytest colorbench/tests/

# 5) Karşılaştırma raporu:
python colorbench/scripts/category_report.py
```

### 6.2 Otomatik olan ne?

- 81 measure_* fonksiyonu otomatik koşar (~2-5 dakika)
- 1666 flat metric snapshot'a kaydedilir
- `comparison.py` automatic side-by-side scorecard
- HTML report otomatik
- Pin test auto-runs (snapshot regression)
- `category_report.py` 4 meta kategoride side-by-side matrix

### 6.3 Tahmini efor

**Sade Bjorn-paradigması bir uzay** için ~30-60 dakika:
- Class implementasyonu: 15-30 dk (matematik basit ise)
- run.py register: 1 dk
- ColorBench koşumu: 2-5 dk
- Comparison rapor: anında

---

## 7. Vurgun eşiği — yeni uzay ne zaman gerçekten "iyi"?

| Eşik | Anlam |
|---|---|
| `perceptual_visible` solo win > 30% | Tasarımcının görebileceği gerçek fark |
| `tied` < 50% | Sub-JND oyunu değil, hakiki ayrım |
| He 2022 STRESS < 33 | Generalization (overfit değil) |
| `mathematical` round-trip < 1e-12 | Numerik sağlamlık |
| 81 measure_*'tan en az 50'sinde **WIN ya da TIE** | Bütünsel rakip yok demek |

**Helmlab'ın şu anki perceptual_visible:** ~28% solo win, %55 tied. Eşik geçilmemiş. Bu yüzden "büyük vurgun" değil, "niche specialist" pozisyonunda.

**Bjorn'un OKLab paradigması:** sadelik kazandı (3 stage, 1 transfer, 1 LMS matrisi). Bizim yaklaşımımız (15+ stage, çoklu CMA-ES) diminishing returns'e takıldı. Yeni uzayda **sadelik > parametre sayısı**.

---

## 8. Ne yapmalı, ne yapmamalı

### Yap

- ✅ Sade uzay tasarla (3-5 stage max)
- ✅ Her stage'i analitik invertable tut (Newton iter sadece son çare)
- ✅ Phase 11 fair test mantığını bozma (CIE Lab hedef = parametre-semantik fix)
- ✅ He 2022 holdout'unu rapor et (eğitime sokma)
- ✅ Snapshot regression PASS bırak
- ✅ Yeni `measure_user_*` eklerken → `metric_categories.py`'a kayıt + pin test'e ekle

### Yapma

- ❌ Test-space koordinatı (L=0.6) hedef olarak kullanma → CIE Lab hedef kullan (L*=60, C*=30)
- ❌ HumFB self-collected üzerinde "kazandık" deme — kültürel bias var
- ❌ Sub-JND kazanımları "vurgun" olarak sun — `jnd_aware_summary` ile dürüstçe ayır
- ❌ Pipeline'a parametre eklemekten "iyileşme" bekleme — diminishing returns kanıtlanmış
- ❌ Lab-cartesian olmayan paradigma (LCh polar, HSL) test ediyorsan → ColorBench'in cartesian-bias'ını rapora yaz

---

## 9. Hızlı referans — komutlar

```bash
# Tek uzay koş
python colorbench/run.py oklab

# İki uzay side-by-side
python colorbench/run.py oklab mynew

# Sadece bir kategori
python colorbench/run.py oklab --category perceptual_visible

# Canonical literatür uzayları
python colorbench/run.py oklab --canonical

# Measurement pipeline
python colorbench/run.py metric --json my_params.json

# Pin testler
pytest colorbench/tests/

# Kategori bazlı karşılaştırma raporu
python colorbench/scripts/category_report.py
```

---

## 10. Özet

ColorBench artık 81 fonksiyon × 1666 flat metric × 32+ pin test ile **production-ready**. Yeni bir uzay denerken bu doc'u rehber olarak tut. Hedef: `perceptual_visible` solo win > %30 + He 2022 holdout PASS + sadelik. Bu üçü olmadan "OKLab'ı yendik" iddiası yapma.
