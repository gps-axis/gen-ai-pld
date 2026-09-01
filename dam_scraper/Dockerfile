FROM mcr.microsoft.com/playwright/python:v1.61.0-noble

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY auth_session.py dam_auth.py dam_scrape.py ./

# The Playwright image supplies this unprivileged browser user. Keep writable
# session data outside the application directory so the code remains read-only.
RUN mkdir -p /home/pwuser/.dam-auth \
    && chown -R pwuser:pwuser /home/pwuser/.dam-auth

USER pwuser
ENV DAM_AUTH_STATE=/home/pwuser/.dam-auth/state.json

ENTRYPOINT ["python", "dam_scrape.py"]
