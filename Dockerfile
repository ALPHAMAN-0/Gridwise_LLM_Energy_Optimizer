# GridWise API image.
#
# Debian slim on purpose, NEVER alpine: the CBC solver binary bundled inside the
# PuLP wheel is dynamically linked against glibc and libstdc++, and will not
# execute on musl.
FROM python:3.13-slim

# TMPDIR: PuLP writes temporary .mps/.sol files per solve, and the process runs
# as a non-root user, so the temp directory must be one that user can write to.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TMPDIR=/tmp \
    PORT=8000

WORKDIR /app

# Dependencies first so this layer stays cached while application code changes.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code only. Never `COPY . .`: the build context may hold a local
# .env, and secrets must reach the container as runtime env vars, not layers.
COPY app ./app

RUN useradd -m -u 1000 appuser
USER 1000

# Build-time smoke solve, run as the runtime user: if the bundled CBC binary
# cannot execute here (wrong libc, wrong arch, unwritable TMPDIR) the build
# fails now instead of the first request failing in production.
RUN python -c "import pulp; p = pulp.LpProblem('smoke', pulp.LpMinimize); x = pulp.LpVariable('x', lowBound=0); y = pulp.LpVariable('y', lowBound=0); p += 2 * x + 3 * y; p += x + y >= 4; p += x <= 3; p.solve(pulp.PULP_CBC_CMD(msg=False)); status = pulp.LpStatus[p.status]; assert status == 'Optimal', status; assert abs(pulp.value(p.objective) - 9.0) < 1e-6, pulp.value(p.objective); print('CBC smoke solve:', status)"

EXPOSE 8000

# slim ships no curl, so the probe is plain urllib.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, sys, urllib.request; r = urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/health', timeout=4); sys.exit(0 if r.status == 200 else 1)"

# One worker on purpose: the interpretation cache and the warm-up solve live in
# process memory, and the free instance has 512 MB. `sh -c` so that the PORT
# Render injects at runtime is honoured; `exec` so uvicorn is PID 1 and
# receives SIGTERM directly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
