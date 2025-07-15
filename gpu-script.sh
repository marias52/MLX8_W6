#!/usr/bin/env bash

missing_vars=()

# Check each required variable
[[ -z "$WANDB_API_KEY" ]] && missing_vars+=("WANDB_API_KEY")
[[ -z "$GITHUB_NAME" ]]   && missing_vars+=("GITHUB_NAME")
[[ -z "$GITHUB_EMAIL" ]]  && missing_vars+=("GITHUB_EMAIL")

if (( ${#missing_vars[@]} > 0 )); then
    echo -e "\nThe following required environment variables are NOT set:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo -e "\nPlease export them before proceeding.\n"
    exit 1
else
    echo -e "\n✅ All required environment variables are set. Continuing...\n"
fi

git config --global user.name "$GITHUB_NAME"
git config --global user.email "$GITHUB_EMAIL"

apt update
apt install -y vim rsync git git-lfs nvtop htop tmux curl btop

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# starship
curl -sS https://starship.rs/install.sh | sh
echo 'eval "$(starship init bash)"' >> ~/.bashrc

mkdir -p "~/.config"
cat > ~/.config/startship.toml <<EOF
[directory]
truncation_length = 3
truncate_to_repo = false
fish_style_pwd_dir_length = 1
home_symbol = "~"
EOF

# duckdb
# curl https://install.duckdb.org | sh
# echo "export PATH='/root/.duckdb/cli/latest':\$PATH" >> ~/.bashrc
# source ~/.bashrc

uv sync
# activate virtual environment for running python scripts
source .venv/bin/activate
echo "Setup complete - virtual environment activated. You can now run Python scripts directly."
echo "Run 'git lfs pull' to download large files."

which python
which uv

tmux