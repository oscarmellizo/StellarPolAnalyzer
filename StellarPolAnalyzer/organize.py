import os
import shutil
from pathlib import Path
from collections import defaultdict
from astropy.io import fits

def organize_night_images(
    night_dir,
    output_base,
    target_header="OBJECT",
    filter_header="INSFLNAM",
    angle_header="ANGLE",
    file_pattern="*.fits"
):
    """
    Agrupa las FITS de una noche en series de 4 (una por cada ángulo de polarización),
    copiándolas a carpetas organizadas por target y filtro.

    Para cada fichero:
      • Intenta leer ANGLE; si no existe, toma la 2ª palabra de OBJECT como ángulo.
      • El primer token de OBJECT será el 'target' (sin espacios).
      • INSFLNAM será el 'filter'.
    Luego agrupa por (target, filter) y valida que haya 4 ángulos distintos.
    Copia cada grupo en output_base/target/filter/.

    Parameters
    ----------
    night_dir : str or Path
        Carpeta con todas las FITS de la noche.
    output_base : str or Path
        Carpeta donde crear subcarpetas target/filter.
    target_header : str
        Header FITS con "OBJECT" por defecto.
    filter_header : str
        Header FITS con "INSFLNAM" por defecto.
    angle_header : str
        Header FITS con "ANGLE" por defecto.
    file_pattern : str
        Patrón glob para localizar FITS (p.ej. "*.fits").

    Returns
    -------
    groups : list of dict
        Cada dict contiene:
          - 'target'   : str
          - 'filter'   : str
          - 'angles'   : sorted list de 4 floats
          - 'paths'    : dict ángulo→Path
          - 'group_dir': Path al directorio creado
    Raises
    ------
    ValueError
        Si para algún (target,filter) no se encuentran exactamente 4 ángulos únicos.
    """
    night_dir   = Path(night_dir)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    # Map: (target,filter) -> {angle: filepath}
    index = defaultdict(dict)

    for fpath in night_dir.glob(file_pattern):
        try:
            hdr = fits.getheader(fpath, 0)
        except Exception:
            continue

        # Filtrar sólo si tiene filtro
        flt = hdr.get(filter_header)
        if flt is None:
            continue

        # OBJECT → target (+ fallback para ANGLE)
        obj = hdr.get(target_header, "").strip()
        if not obj:
            continue
        parts = obj.split()
        tgt = parts[0]

        # Leer ANGLE o fallback
        ang = hdr.get(angle_header)
        if ang is None:
            if len(parts) >= 2:
                try:
                    ang = float(parts[1])
                except ValueError:
                    continue
            else:
                continue
        else:
            ang = float(ang)

        # Indexar
        index[(tgt, flt)][ang] = fpath

    groups = []
    for (tgt, flt), ang_map in index.items():
        angles = sorted(ang_map.keys())
        if len(angles) != 4:
            raise ValueError(
                f"Para target={tgt}, filter={flt} se hallaron ángulos: {angles}, "
                "pero se requieren exactamente 4."
            )

        # Crear carpeta target/filter
        group_dir = output_base / tgt / flt
        group_dir.mkdir(parents=True, exist_ok=True)

        paths = {}
        for ang in angles:
            src = ang_map[ang]
            dst = group_dir / src.name
            shutil.copy2(src, dst)
            paths[ang] = dst

        groups.append({
            "target":    tgt,
            "filter":    flt,
            "angles":    angles,
            "paths":     paths,
            "group_dir": group_dir
        })

    return groups
