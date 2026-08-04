# Local release checklist

This repository has **not** been released, uploaded, archived or assigned a DOI.
Everything below is outstanding and requires a human decision. None of it has been
performed, and none of it can be performed by an automated pass.

## Blocking items

### 1. Licence — RESOLVED (MIT, 2026-08-04)

The repository is released under the MIT Licence, © 2026 Tilock Sadhukhan.
See `../LICENSE`.

Still to confirm, if not already done:

- [ ] Institutional approval (CDAC and any collaborating institution)
- [ ] Confirmation that no funder agreement constrains the choice
- [ ] Confirmation of copyright ownership, and that all co-authors agree to MIT

If any of these turns out to conflict, change the licence **before** the
repository is made public — relicensing after public release is far harder.

### 2. Author and affiliation metadata — BLOCKING

`CITATION.cff` currently contains only what could be verified from the repository
itself (the git author name and email). Still required:

- [ ] Full author list with the agreed ordering
- [ ] Affiliations
- [ ] ORCID identifiers — **must not be invented**; each must be supplied by its
      owner
- [ ] Corresponding-author designation
- [ ] Funding acknowledgements and grant numbers

### 3. Publication metadata — BLOCKING until acceptance

- [ ] Journal name, volume, pages
- [ ] Publication DOI
- [ ] Acceptance and publication dates

### 4. Archive and repository — BLOCKING

- [x] Repository created 2026-08-04 and made **public** the same day, MIT-licensed
- [x] Repository URL added to `CITATION.cff` and the README
- [ ] Decide on an archive (Zenodo, institutional repository, or none)
- [ ] Obtain an archive DOI **after** deposit, not before
- [ ] Replace the repository-URL and DOI placeholders in `CITATION.cff`, the
      README and `docs/MANUSCRIPT_ALIGNMENT.md`

## Verification items

- [ ] `pytest` passes from a clean environment
- [ ] `python scripts/reproduce.py --profile paper` completes
- [ ] `python scripts/verify_results.py --full` passes
- [ ] Every figure inspected at final size for clipped labels, tick collisions and
      legibility
- [ ] Every table checked for units, precision and agreement with
      `paper_values.tex`
- [ ] `docs/MANUSCRIPT_ALIGNMENT.md` worked through item by item

## Bibliography — RESOLVED (2026-08-04)

- [x] All 19 entries verified against the published record (publisher pages,
      arXiv, NASA ADS). Volumes, pages, years, DOIs and ISBNs checked.
- [ ] `somma2015` — confirm no journal version has appeared since the 2016
      revision; otherwise cite the preprint as written.
- [ ] `pandas` — pin the Zenodo DOI to the exact release used (3.0.2), or switch
      to the concept DOI, per your preferred house style.

The two open items are style decisions, not unverified facts. No entry is a
placeholder and none was invented.

## Data availability statement

Draft wording, to be used only once the items above are resolved:

> The code and configuration files used to generate all numerical results and
> figures in this manuscript are available at [REPOSITORY URL], archived at
> [ARCHIVE DOI]. All figures and tables can be regenerated with
> `python scripts/reproduce.py --profile paper`.

Both placeholders must be filled with real values before this statement is used.
Until then it is inaccurate.

## Explicitly not done

No push, tag, release, pull request, issue, package publication or upload to any
external service has been performed. Every artefact described here exists only in
the local working tree.
