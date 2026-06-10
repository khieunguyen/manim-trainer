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
    .add_local_file(
        "setup_env.sh",
        "/root/setup_env.sh",
        copy=True
    )
    .run_commands(
        "bash /root/setup_env.sh"
    )
)

app = modal.App("manim-trainer")


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
)
def train_manim():
    import subprocess
    subprocess.run(["bash", "/data/manim-trainer/scripts/train_manim.sh"], check=True)
    volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={"/data": volume},
)
def train_skip_sft():
    import subprocess
    subprocess.run(["bash", "/data/manim-trainer/scripts/train_manim_skip_sft.sh"], check=True)
    volume.commit()

@app.local_entrypoint()
def main(task: str = "train_manim"):
    if task == "setup":
        setup.remote()
    elif task == "train_manim":
        train_manim.remote()
    elif task == "train_skip_sft":
        train_skip_sft.remote()
    else:
        raise ValueError(f"Unknown task: {task}")
