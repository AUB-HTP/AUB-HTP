# Installation

## pip (standard)

```bash
pip install aub-htp
```

## apt (Debian/Ubuntu)

(Work in Progress)
```bash
sudo apt install python3-aub-htp
```

## uv

[Install uv](https://docs.astral.sh/uv/), then:

```bash
uv init myproject && cd myproject
uv add aub-htp
```

## conda

[Install Miniconda](https://docs.conda.io/en/latest/miniconda.html), then:

```bash
conda install -c conda-forge aub-htp
```

## poetry

[Install Poetry](https://python-poetry.org/docs/), then:

```bash
poetry new myproject && cd myproject
poetry add aub-htp
```

## Development install

```bash
git clone https://github.com/AUB-HTP/AUB-HTP
cd AUB-HTP
pip install .
```
