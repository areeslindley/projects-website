# Literary NLP data

Page-windowed public-domain novels used by the notebooks.

## Bundled file

- `pages.csv` — one row per ~250-word page, with book metadata, chapter heading, page index, progress (0–1), word count, and text.

Notebooks read **only** this CSV so GitHub Actions does not fetch Project Gutenberg at build time.

## Rebuild from Gutenberg

From the repository root (requires network):

```bash
python projects/literary-nlp/_build_data.py
```

Raw downloads are cached under `data/raw/` (not committed). Texts are stripped of Project Gutenberg header/footer boilerplate before pagination.

## Attribution

All eight titles are public-domain works distributed by [Project Gutenberg](https://www.gutenberg.org/). Project Gutenberg is a trademark of the Project Gutenberg Literary Archive Foundation. This analysis is an independent educational use of the plain-text ebooks.

| Title | Author | Gutenberg ID |
| --- | --- | --- |
| Dracula | Bram Stoker | 345 |
| Wuthering Heights | Emily Brontë | 768 |
| The Time Machine | H. G. Wells | 35 |
| Pride and Prejudice | Jane Austen | 1342 |
| Frankenstein | Mary Shelley | 84 |
| Alice's Adventures in Wonderland | Lewis Carroll | 11 |
| A Christmas Carol | Charles Dickens | 46 |
| The Picture of Dorian Gray | Oscar Wilde | 174 |
