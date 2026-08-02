FROM python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/EnzoCanonero/count-significance"
LABEL org.opencontainers.image.authors="Enzo Canonero <Enzo.Canonero@rhul.ac.uk>; Glen Cowan <G.Cowan@rhul.ac.uk>"

ENV MPLBACKEND=Agg

WORKDIR /app

COPY pyproject.toml PYPI_README.md LICENSE ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir . matplotlib \
    && python -c "import count_significance; import matplotlib"

COPY scripts/make_*_plots.py ./scripts/
COPY config/paper_*.yaml ./config/

RUN mkdir -p plots

CMD ["python"]
