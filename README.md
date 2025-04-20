# 🌟 StellarPolAnalyzer

> **Empareja, fotometriza y astrometriza imágenes polarimétricas de campo estelar en un solo flujo.**

---

## 📚 Descripción

**StellarPolAnalyzer** es una librería Python que automatiza un completo **pipeline polarimétrico** sobre imágenes FITS astronómicas:

1. 🔭 **Detección** de estrellas con `DAOStarFinder`.  
2. 🤝 **Emparejado** de componentes ordinaria/extraordinaria (beam‑splitter).  
3. 📈 **Fotometría** de apertura en 4 ángulos de polarización.  
4. ➗ Cálculo de parámetros de polarización: **Q, U, P, θ** y sus errores.  
5. 🖼️ **Alineación** de imágenes (cross‑correlation).  
6. ✨ **Imagen sintética** y solución WCS con Astrometry.Net.  
7. 🌐 **Cruce SIMBAD** para identificar objetos celestes.  
8. 🎨 **Visualización** clara de pares, aperturas y resultados.  
9. 📑 **Reporte** modular o flujo completo “one‑click”.  

---

## 🚀 Instalación

```bash
pip install StellarPolAnalyzer
```

> Requiere Python ≥3.7 y dependencias:
> `numpy`, `astropy`, `photutils`, `scikit-learn`, `scikit-image`, `scipy`, `astroquery`, `matplotlib`.

---

## 📁 Estructura del paquete

```
StellarPolAnalyzer/
├─ alignment.py            # Alineación FITS y guardado con cabeceras
├─ detection.py            # Detección de fuentes y process_image
├─ pairing.py              # Cálculo de distancias/ángulos y emparejado
├─ photometry.py           # Aperture photometry y polarimetría
├─ astrometry.py           # Imagen sintética, WCS y SIMBAD
├─ visualization.py        # Funciones de dibujo y guardado de plots
├─ pipeline.py             # compute_full_polarimetry + run_complete_pipeline
├─ utils.py                # Utilidades (exportar pares a TXT)
└─ __init__.py             # API pública
```

---

## 🛠️ Uso detallado

### 1️⃣  Detección y emparejado básico

```python
from StellarPolAnalyzer.detection import process_image
from StellarPolAnalyzer.visualization import draw_pairs

# 1. Procesar una imagen FITS (p.ej. 0°)
img_path = 'data/field_0.fits'
data, sources, cands, pairs, d_mode, a_mode = process_image(
    img_path,
    fwhm=3.0,
    threshold_multiplier=5.0,
    tol_distance=1.44,
    tol_angle=1.20,
    max_distance=75
)

# 2. Visualizar resultados
draw_pairs(
    image_data=data,
    sources=sources,
    pairs=pairs,
    num_stars=len(sources),
    mode_distance=d_mode,
    mode_angle=a_mode,
    tol_distance=1.44,
    tol_angle=1.20,
    original_name='field_0.fits',
    filename_suffix='_pairs',
    report_dir='reports/basic'
)
```

> 🔎 **Salida**:  
> - Gráfico con puntos rojos (centroides), líneas lime (parejas),  
> - Círculos azules/rojos marcando “ordinaria” y “extraordinaria”,  
> - Leyenda resumen fuera del mapa.

---

### 2️⃣  Fotometría y cálculo de polarización por pares

```python
from StellarPolAnalyzer.photometry import compute_polarimetry_for_pairs

final_paths = [
    'data/field_0-aligned.fits',
    'data/field_22.5-aligned.fits',
    'data/field_45-aligned.fits',
    'data/field_67.5-aligned.fits',
]
# `sources` y `pairs` obtenidos de process_image/ref_image
results = compute_polarimetry_for_pairs(
    final_image_paths=final_paths,
    sources=sources,
    final_pairs=pairs,
    aperture_radius=5,
    r_in=7,
    r_out=10,
    SNR_threshold=5
)

# Ejemplo de salida:
for entry in results:
    print(f"Par {entry['pair_index']:02d}: q={entry['q']:.2f}%  u={entry['u']:.2f}%  P={entry['P']:.2f}%  θ={entry['theta']:.1f}°")
```

---

### 3️⃣  Pipeline polarimétrico (4 imágenes)

```python
from StellarPolAnalyzer.pipeline import compute_full_polarimetry

ref = 'data/field_0.fits'
others = ['data/field_22.5.fits', 'data/field_45.fits', 'data/field_67.5.fits']
proc, polar_results, aligned_paths = compute_full_polarimetry(
    ref_path=ref,
    other_paths=others,
    fwhm=3.0,
    threshold_multiplier=5.0,
    tol_distance=1.44,
    tol_angle=1.20,
    max_distance=75,
    phot_aperture_radius=5,
    r_in=7,
    r_out=10,
    SNR_threshold=5,
    save_plots=True,
    report_dir='reports/full_pipeline'
)

print("Imágenes alineadas:", aligned_paths)
print("Número de pares analizados:", len(polar_results))
```

---

### 4️⃣  Pipeline completo + astrometría + SIMBAD

```python
from StellarPolAnalyzer.pipeline import run_complete_polarimetric_pipeline
import astropy.units as u

ref = 'data/field_0.fits'
others = ['data/field_22.5.fits', 'data/field_45.fits', 'data/field_67.5.fits']
angles = [0.0, 22.5, 45.0, 67.5]

final_paths, polar_results, wcs, enriched = run_complete_polarimetric_pipeline(
    ref_path=ref,
    other_paths=others,
    pol_angles=angles,
    fwhm=3.0,
    threshold_multiplier=5.0,
    tol_distance=1.44,
    tol_angle=1.20,
    max_distance=75,
    phot_aperture_radius=5,
    r_in=7,
    r_out=10,
    SNR_threshold=5,
    astrometry_api_key='TU_API_KEY_ASTRONET',
    simbad_radius=0.01*u.deg,
    synthetic_name='synthetic_field.fits',
    save_plots=True,
    report_dir='reports/complete'
)

# Resultados finales
print("🔭 FITS usados:", final_paths)
print("⭐ Polarimetría + SIMBAD:")
for entry in enriched:
    print(f"Par {entry['pair_index']:02d} → P={entry['P']:.2f}%  θ={entry['theta']:.1f}°  Obj={entry['simbad_id']}")
```

---

### 5️⃣  Métodos auxiliares

| Función                                    | Descripción                                    |
|--------------------------------------------|------------------------------------------------|
| `detect_stars(image_data, fwhm, thr)`      | Detecta fuentes en `image_data`.               |
| `process_image(path, ...)`                 | Detecta y empareja en 1 imagen.                |
| `compute_distance_angle(p1,p2)`            | Distancia y ángulo mínimo simétrico.           |
| `find_candidate_pairs(sources, max_dist)`  | Todas las parejas dentro de `max_dist`.        |
| `filter_pairs_by_mode(pairs, d_tol, a_tol)`| Filtra por moda de distancia/ángulo.           |
| `align_images(ref,img)`                    | Alinea `img` a `ref` (cross-correlation).      |
| `save_fits_with_same_headers(...)`         | Guarda FITS manteniendo su cabecera original.  |
| `draw_pairs(...)`                          | Visualiza pares y leyenda, guarda PNG.         |
| `save_plot(...)`                           | Guarda imagen genérica en PNG.                 |
| `compute_polarimetry_for_pairs(...)`       | Fotometría + Q/U/P/θ.                          |
| `compute_full_polarimetry(...)`            | Pipeline polarimetría (4 imágenes).            |
| `run_complete_polarimetric_pipeline(...)`  | Pipeline completo + astrometría + SIMBAD.      |

---

## 📄 Licencia

**MIT License** – ¡Uso libre, modifícalo y contribuye! 👐

---

## 🤝 Contribuciones

Pull requests, issues y ⭐️ son siempre bienvenidos. ¡Ayuda a mejorar la ciencia abierta! 🚀

---

*¡Lleva tu análisis polarimétrico un paso más allá!* ✨
