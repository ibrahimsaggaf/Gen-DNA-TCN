CHAR_TO_CODE = {'A': 0, 'C': 1, 'G': 2, 'N': 3, 'T': 4}
CODE_TO_CHAR = {0: 'A', 1: 'C', 2: 'G', 3: 'N', 4: 'T'}
NUM_CHAR = len(CHAR_TO_CODE)
LEFT_ADAPTER = [CHAR_TO_CODE[char] for char in 'TGCATTTTTTTCACATC']
RIGHT_ADAPTER = [CHAR_TO_CODE[char] for char in 'GGTTACGGCTGTT']