# Releasing gang-mof

`pip install "gang-mof @ git+https://github.com/MehrdadJalali-AI/GANg"` works as soon as the code is on `main`.
To make the shorter `pip install gang-mof` work, publish to PyPI once:

1. Create an account at <https://pypi.org> and enable two-factor authentication.
2. On PyPI go to *Your account → Publishing → Add a new pending publisher* and enter
   project `gang-mof`, owner `MehrdadJalali-AI`, repository `GANg`, workflow `publish.yml`, environment `pypi`.
3. On GitHub go to *Settings → Environments → New environment* and create `pypi`.
4. Bump `src/gang_mof/_version.py` (and `CITATION.cff`), commit, then create a GitHub release with tag `v0.1.0`.
   The `publish to PyPI` workflow builds the wheel and uploads it.

Optional: enable the Zenodo GitHub integration before creating the release to get a citable DOI for each version.
