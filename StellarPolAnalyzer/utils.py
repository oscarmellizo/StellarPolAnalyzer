def write_candidate_pairs_to_file(candidate_pairs, filename="candidate_pairs.txt"):
    """
    Escribe las parejas candidatas de estrellas en un archivo de texto tabulado.

    Cada línea del archivo contendrá cuatro columnas separadas por tabuladores:
      - Estrella_A: índice de la primera estrella de la pareja.
      - Estrella_B: índice de la segunda estrella de la pareja.
      - Distancia_px: separación entre ambas estrellas en píxeles, formateada a dos decimales.
      - Ángulo_deg: ángulo de emparejamiento en grados, formateado a dos decimales.

    Parámetros
    ----------
    candidate_pairs : list of tuple
        Lista de tuplas (i, j, distance, angle), donde:
          i (int)       – índice de la primera estrella,
          j (int)       – índice de la segunda estrella,
          distance (float) – distancia en píxeles,
          angle (float)    – ángulo en grados.
    filename : str, opcional
        Ruta y nombre del archivo de salida. Por defecto "candidate_pairs.txt".

    Ejemplo
    -------
    >>> pairs = [(0, 1, 36.37, 179.76), (2, 3, 36.37, 179.76)]
    >>> write_candidate_pairs_to_file(pairs, "pares.txt")
    # Crea 'pares.txt' con encabezado y los datos formateados.
    """
    with open(filename, 'w') as f:
        f.write("Estrella_A\tEstrella_B\tDistancia_px\tÁngulo_deg\n")
        for i, j, d, a in candidate_pairs:
            f.write(f"{i}\t{j}\t{d:.2f}\t{a:.2f}\n")
