def write_candidate_pairs_to_file(candidate_pairs, filename="candidate_pairs.txt"):
    """Save candidate pairs to tab-delimited file."""
    with open(filename,'w') as f:
        f.write("Estrella_A\tEstrella_B\tDistancia_px\tÁngulo_deg\n")
        for i,j,d,a in candidate_pairs:
            f.write(f"{i}\t{j}\t{d:.2f}\t{a:.2f}\n")
