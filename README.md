# Algorithms for genomic data analysis (Assignment 1)
*Winter semester 2023/2024*
**Simple sequence aligner written in pure Python**

## Task defition

Implement a read mapping algorithm that

- is designed to work on reads of `length∼ 1 kbp` with error rate `5 − 10%`,
- may fail to map some reads, but cannot return incorrect alignments,
- is efficient (i.e. fast) and has good quality (i.e. high proportion of mapped reads).

In your program you can use:
- the codes from the classes,
- libraries included in standard Python distribution, as well as NumPy, SciPy and Biopython,
- available code for building suffix arrays or similar structures (e.g. the attached implementation of the
    Karkkainen-Sanders algorithm).

In your program you cannot use:
- programs and libraries to read assembly, mapping, alignment, etc.
- multiprocessing commands,
- subprograms written in other languages.

The solution should include:
- program file written in Python 3,
- slides with short description of your approach.

### Specification and attached files

Minimum performance requirements are following:
- Given the reference sequence of `length≤20 Mbp`, a collection
of rreads should be processed in at most `2+r/10 minutes` and `memory<1GB`, resulting
in `≥80%` reads mapped and no read mapped incorrectly.

Program should be executable using syntax:
`python3 mapper.py reference.fasta reads.fasta output.txt`

Input data are in fasta format. Output file should consist of one line for each mapped read, consisting of read
identifier, start and end positions of read alignment, separated by tabs.

It can be assumed that all reads come from the reference sequence, but contain errors (substitutions, in-
sertions and deletions of single nucleotides) resulting from the sequencing process. Errors occur independently
at each position at the assumed error rate, so the total number of errors may slightly exceed10%of the read
length. A read is mapped correctly when the mapping coordinates differ from the actual coordinates of the
fragment it comes from by `≤20 bp`.

## Solution

### Running the aligner

The solution uses [Poetry as a package manager](https://python-poetry.org/docs/), logging with custom colouring and other non-computation-specific utility libraries.
The entry script should install Poetry by default when you run it via the original syntax i.e.:
```bash
    python3 mapper.py data/reference.fasta data/reads0.fasta output.txt
```

In case the script fails to install [Poetry](https://python-poetry.org/docs/) (note that this was tested on the university server already), you can manually install dependencies and run the script:
```bash
    # Install poetry
    $ pipx install poetry
    # Install dependencies
    $ poetry install
    # Run the aligner
    $ poetry run aadg_genomics_class/cli.py data/reference.fasta data/reads0.fasta output.txt
```

You can pass additional parameters to the aligner. Please use `--help` parameter to display help information:
```bash
    $ python3 mapper.py --help
    # or alternative syntax:
    $ poetry run aadg_genomics_class/cli.py --help
```

The help can look like this:

![CLI usage screenshot](https://github.com/styczynski/aadg-genomics-class/blob/main/static/screen0.png?raw=true)

### Alignment algorithms introduction

Sequence alignment is a transformative process that elucidates the steps required to derive one sequence from another. In the realm of bioinformatics, its primary application lies in identifying analogous segments within DNA chains, RNA chains, or proteins to unveil evolutionary and functional associations. The approach commonly employed for such alignments is dynamic programming, a methodology that resolves intricate problems by dissecting them into smaller, recursive subproblems. The outcomes of these subproblems are stored and utilized to reconstruct the ultimate solution.

Various versions of pairwise alignment algorithms exist, such as the Needleman-Wunsch algorithm for global alignment, the Smith-Waterman algorithm for local alignment, and semi-global algorithms tailored for suffix-prefix and prefix-suffix alignments. Distinctions among them primarily lie in the initialization step and the starting point for the backtrack procedure.

Given the quadratic time complexity of alignment algorithms, the expedited detection of similar regions between sequences often employs k-mer indexing. However, the comprehensive collection of all k-mers can strain computational resources, particularly when targeting frequently occurring k-mers in the sequence set. Focusing on a subset of k-mers can mitigate these challenges while maintaining a reasonable level of sensitivity.

One such approach involves leveraging lexicographically smallest k-mers known as minimizers, [as detailed here](https://academic.oup.com/bioinformatics/article/20/18/3363/202143).

### High-level algorithm of the aligner

**The current implemenation works (on the highest-level) somewhere like this:**

1. Load all target and query sequences and covert them to numpy arrays
2. I build a minimizer index from the target sequence, which will store all positions and origins for each distinct minimizer found in the reference
3. Ignore too frequent minimizers should (controlled by parameter) in that target index
4. For each query
    1. Build a minimizer index for that sequence
    2. All minimizers of a query are to be searched against the reference index to find matches. From the list of all matches for a pair of (query, reference), the longest linear chain should represent the best candidate for a good alignment between the pair. That region can be obtained in quasilinear time by solving the longest increasing subsequence problem on the list of minimizer matches.
    3. I invoke aligner (Needleman-Wunsch) only on the found regions
    4. I print matched regions with correct position paddings
5. Print summary reports
