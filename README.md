# 🌟 StellarPolAnalyzer

**_Empareja, fotometriza y astrometriza imágenes polarimétricas de campo estelar en un solo flujo._**

----

## 📚 Descripción

StellarPolAnalyzer es una **librería Python** que automatiza todo el **pipeline polarimétrico** en imágenes astronómicas:

1. **Detección** de estrellas con DAOStarFinder.
2. **Emparejado** de componentes ordinaria/extraordinaria (beam splitter).
3. **Fotometría** de apertura en 4 ángulos de polarización.
4. Cálculo de parámetros de polarización: **Q, U, P, θ** y errores.
5. **Alineación** de imágenes (phase cross-correlation).
6. **Generación de imagen sintética** y solución WCS con Astrometry.Net.
7. **Cruce SIMBAD** para identificar objetos celestes.
8. **Visualización** clara de pares, aperturas y resultados.
9. **Reporte rápido** vía funciones modulares o pipeline completo.

Ideal para tus proyectos de **TFM**, exoplanetas, discos circumestelares o estudios de polarización galáctica.

----

## 🚀 Instalación

```bash
pip install StellarPolAnalyzer
```

> Requiere Python ≥3.7 y dependencias:
> `numpy`, `astropy`, `photutils`, `scikit-learn`, `scikit-image`, `scipy`, `astroquery`, `matplotlib`.

----

## 📁 Estructura del paquete

```text
StellarPolAnalyzer/
├─ alignment.py        # Alineación FITS y guardado
├─ detection.py        # Detección de fuentes y process_image
├─ pairing.py          # Cálculo de distancias/ángulos y emparejado
├─ photometry.py       # Aperture photometry y polarimetría
├─ astrometry.py       # Imagen sintética, Astrometry.Net y SIMBAD
├─ visualization.py    # Funciones de dibujo y resumen
├─ pipeline.py         # compute_full_polarimetry + run_complete_pipeline
├─ utils.py            # Utilidades (exportar pares)
└─ __init__.py         # API pública
```

----

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
    tol_angle=1.20
)
```

> 🔎 **Salida**: gráfico con puntos rojos (estrella), líneas lime (parejas), círculos azules/rojos y resumen.

----

### 2️⃣ Pipeline completo de polarimetría + astrometría

```python
from StellarPolAnalyzer.pipeline import run_complete_polarimetric_pipeline
import astropy.units as u

# Definir paths y ángulos
ref = 'data/field_0.fits'
others = ['data/field_22.fits', 'data/field_45.fits', 'data/field_67.fits']
angles = [0.0, 22.5, 45.0, 67.5]

# Ejecutar flujo completo
final_paths, polar_results, wcs, enriched = \
    run_complete_polarimetric_pipeline(
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
        astrometry_api_key='TU_API_KEY',
        simbad_radius=0.01*u.deg,
        synthetic_name='synthetic_field.fits'
    )

# Mostrar resultados básicos
print('°° Imágenes procesadas:')
for p in final_paths:
    print(' -', p)

print('\n⭐ Polarimetría por par:')
for entry in enriched:
    print(f"Par {entry['pair_index']:02d} -> P={entry['P']:.2f}%  θ={entry['theta']:.1f}°  SIMBAD={entry['simbad_id']}")
```

> 🌐 **Salida**:
> - FITS alineados
> - Lista de dicts con `{pair_index, q, u, P, θ, error, ra, dec, simbad_id}`

----

### 3️⃣ Métodos individuales

| Función                                    | Descripción                                    |
|--------------------------------------------|------------------------------------------------|
| `detect_stars(image_data, fwhm, thr)`      | Detecta fuentes en `image_data`.               |
| `process_image(path, ...)`                 | Detección + emparejado en 1 imagen.            |
| `compute_distance_angle(p1, p2)`          | Distancia y ángulo minimal entre 2 puntos.     |
| `find_candidate_pairs(sources, max_dist)`  | Todas las parejas en rango `max_dist`.         |
| `filter_pairs_by_mode(pairs, tol_d, t_a)`  | Filtra por distancia/ángulo modal.             |
| `compute_polarimetry_for_pairs(...)`      | Fotometría + Q/U/P/θ en 4 ángulos.             |
| `align_images(ref, img)`                   | Alinear `img` a `ref` via phase_cross_correlation. |
| `save_fits_with_same_headers(...)`         | Guarda FITS conservando metadatos original.    |
| `annotate_with_astrometry_net(...)`        | Imagen sintética + WCS + SIMBAD.               |
| `draw_pairs(...)`                          | Visualiza pares y resumen.                     |
| `compute_full_polarimetry(...)`            | Pipeline polarimetría (4 imágenes).            |
| `run_complete_polarimetric_pipeline(...)`  | Pipeline completo + astrometría.               |

----

## 📄 Licencia

Librería **MIT License** – uso libre, modifica y contribuye. 👐

----

## 🤝 Contribuciones

¡Pull requests, issues y estrellas son bienvenidas! 🛠️

----

*¡Lleva tu análisis polarimétrico un paso más allá!* ✨

