# PLD laydown harness.
#
# The container is a CLIENT, not a self-contained system. Three things stay
# outside it and are supplied at run time: the model (QWEN_BASE_URL), the
# segmentation service (SEGMENT_URL), and fal.ai (FAL_KEY). Nothing here
# downloads a model or holds a credential.
#
# Build:  docker build -t pld-harness .
# Run:    see docker-entrypoint.sh, or DOCKER.md
# 3.14, because pyproject.toml pins requires-python to 3.14.* and the host venv
# is 3.14: an image on an older interpreter runs code that was never exercised.
FROM python:3.14-slim

# ImageMagick is no longer called by anything that ships in this image - the
# tools that shelled out to `magick`/`montage` are retired to old/. It stays
# because the two shims below (docker/sips, docker/magick) exist to keep the
# macOS and Linux copies of the source byte-identical, and dropping the package
# would mean the shims lie about what is available. fonts-liberation supplies
# the Arial stand-in below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        imagemagick \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# The source photos are ~45MP (5464x8192). ImageMagick 7 on trixie already
# allows 1GiB/256MP, which clears that comfortably - but `python:*-slim` is a
# moving tag that shipped ImageMagick 6 on bookworm not long ago, and IM6's
# Debian policy caps memory at 256MiB, which pushes every operation on an image
# this size into the disk-backed cache and makes a run crawl for no visible
# reason. Raise it wherever it is, rather than pinning the fix to one version.
RUN set -eux; \
    for p in /etc/ImageMagick-*/policy.xml; do \
        [ -f "$p" ] || continue; \
        sed -i 's/name="memory" value="[^"]*"/name="memory" value="2GiB"/;  \
                s/name="map" value="[^"]*"/name="map" value="4GiB"/;        \
                s/name="area" value="[^"]*"/name="area" value="1GP"/;       \
                s/name="disk" value="[^"]*"/name="disk" value="8GiB"/' "$p"; \
    done

WORKDIR /app

# The venv lives at the path find_python() in harness.py already looks for
# (HERE/.venv/bin/python), so the harness hands the agent an interpreter that
# has numpy/PIL/scipy without any source change. task/SKILL.md also tells the
# agent to run tools with `../.venv/bin/python`, and that resolves here too.
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt ./
RUN /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Two macOS-only binaries the tools shell out to. Shimming them keeps the
# harness source byte-identical to the copy that still runs natively on macOS;
# see the header comment in each file for what they do and why.
COPY docker/sips /usr/local/bin/sips
COPY docker/magick /usr/local/bin/magick.im6
RUN set -eux; \
    chmod +x /usr/local/bin/sips /usr/local/bin/magick.im6; \
    if command -v magick >/dev/null 2>&1; then \
        rm /usr/local/bin/magick.im6; \
    else \
        mv /usr/local/bin/magick.im6 /usr/local/bin/magick; \
    fi; \
    magick -version >/dev/null

# The macOS font path, pointed at a metric-compatible Linux face. common.py's
# contact_sheet() asks PIL for this exact file when it labels the reference-match
# strip, and falls back to DejaVu if it is missing - but the fallback is a
# different face at the same size, so the labels would silently reflow. One line,
# and it keeps the container's output identical to the host's.
RUN mkdir -p /System/Library/Fonts/Supplemental \
    && ln -s /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf \
             /System/Library/Fonts/Supplemental/Arial.ttf

# The reference LIBRARY is deliberately not baked in. It is mounted at
# /in/reference_library and read in place: it is the part that changes, it is
# ~150 MB of an image that otherwise holds five scripts, and a library rebuilt
# into the image is a library nobody can add to without a rebuild.
#
# Its descriptions are cached under /app/.cache/refmatch, so mount a volume
# there (`-v pld-cache:/app/.cache`) or every container re-describes the whole
# library once. Nothing is billed by that; it is time.
COPY harness.py ./
COPY tools/ tools/
COPY task/ task/
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# The path the Kestra flows mount the shared library at. Created empty so a
# run without the mount searches an empty folder and says so, rather than
# failing on a path that does not exist.
RUN mkdir -p /app/library_reference

# The skill used to hardcode the developer's absolute workspace path. It no
# longer does, and this asserts that rather than assuming it: a path that creeps
# back in would send the agent looking outside the container.
RUN ! grep -q '/Users/' task/SKILL.md

# Created empty so a run with no volumes mounted still works; each is a mount
# point in the documented invocation.
RUN mkdir -p /app/inputs /app/runs /app/.cache /in /out

# Unbuffered so `docker logs` shows the agent's turns as they happen rather
# than in one block when the run ends.
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
