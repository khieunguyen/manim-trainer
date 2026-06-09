import modal

volume = modal.Volume.from_name("output", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "pkg-config",
        "build-essential",
        "libcairo2-dev",
        "libpango1.0-dev",
        "texlive-latex-base",
        "texlive-latex-extra",
        "texlive-fonts-recommended",
        "texlive-science",
        "dvipng",
        "tmux",
        "curl",
    )
)

app = modal.App("manim-trainer")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
)
def setup():
    import subprocess
    subprocess.run(["bash", "/data/scripts/setup_env.sh"], check=True)
    volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
)
def train_manim():
    import subprocess
    subprocess.run(["bash", "/data/scripts/train_manim.sh"], check=True)
    volume.commit()


@app.local_entrypoint()
def main(task: str = "train_manim"):
    if task == "setup":
        setup.remote()
    elif task == "train_manim":
        train_manim.remote()
    else:
        raise ValueError(f"Unknown task: {task}")
