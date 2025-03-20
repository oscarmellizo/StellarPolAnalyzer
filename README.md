# StellarPolAnalyzer

StellarPolAnalyzer es una librería de Python para el análisis de imágenes polarimétricas en astronomía. La herramienta permite detectar automáticamente estrellas en imágenes FITS, identificar parejas de estrellas (que corresponden a las dos proyecciones polarimétricas de la misma fuente) y visualizar los resultados. Está pensada para facilitar el análisis de datos polarimétricos y servir como base para desarrollos y análisis adicionales.

## Características

- **Detección de estrellas:**  
  Utiliza DAOStarFinder para detectar estrellas en imágenes FITS con parámetros ajustables (FWHM y threshold).

- **Identificación de parejas:**  
  Emplea un algoritmo basado en NearestNeighbors para encontrar y filtrar parejas de estrellas según la moda de la distancia y el ángulo, usando tolerancias ajustables.

- **Visualización:**  
  Muestra la imagen con:
  - Centros de las estrellas marcados.
  - Líneas que conectan las parejas encontradas.
  - Círculos diferenciados: azul para la estrella de menor X y rojo para la de mayor X.
  - Una leyenda externa con estadísticas (número de estrellas, parejas, etc.).

- **Interfaz gráfica:**  
  Incluye una GUI (basada en Tkinter) para ajustar parámetros y ejecutar el pipeline de forma interactiva.

## Instalación

### Desde PyPI

Una vez publicado, puedes instalar el paquete con:

```bash
pip install StellarPolAnalyzer
```

### Instalación de Desarrollo
Para instalar la versión de desarrollo, clona el repositorio y usa el modo editable:

```bash
git clone https://github.com/oscarmellizo/StellarPolAnalyzer.git
cd StellarPolAnalyzer
pip install -e .
```

## Uso Básico
### Usando la API
La API te permite procesar una imagen FITS y obtener las parejas de estrellas. Por ejemplo:

```python
from StellarPolAnalyzer import process_image, draw_pairs

image_path = 'ruta/a/tu_imagen.fits'
fwhm = 3.0
threshold_multiplier = 5.0
tol_distance = 1.44
tol_angle = 1.20
max_distance = 38.0

# Procesa la imagen
image_data, sources, candidate_pairs, final_pairs, mode_distance, mode_angle = process_image(
    image_path,
    fwhm=fwhm,
    threshold_multiplier=threshold_multiplier,
    tol_distance=tol_distance,
    tol_angle=tol_angle,
    max_distance=max_distance
)

# Visualiza el resultado
draw_pairs(image_data, sources, final_pairs, len(sources), mode_distance, mode_angle, tol_distance, tol_angle)
```

## Contribución
Las contribuciones son bienvenidas. Si deseas colaborar:

1. Haz un fork del repositorio.
2. Crea una rama para tus cambios:
```bash
git checkout -b feature/nueva-funcionalidad
```
4. Realiza tus cambios y envía un Pull Request con una descripción de las modificaciones.

## Licencia
StellarPolAnalyzer se distribuye bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

## Contacto
Para preguntas, sugerencias o reportar problemas, por favor abre un issue en este repositorio o contacta a omellizo@gmail.com.

