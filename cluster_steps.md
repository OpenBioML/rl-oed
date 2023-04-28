### 1. miniconda installation
```bash
cd /fsx/<username>
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p /fsx/<username>/miniconda3
rm -rf miniconda3/miniconda.sh
miniconda3/bin/conda init bash
```

### 2. reconnect

### 3. env setup
```bash
conda env create --prefix /fsx/<username> -f environment.yml
```

### 4. start an interactive session
```bash
srun --account openbioml --partition=g40 --gpus=1 --cpus-per-gpu=12 --job-name=rloed --pty bash -i
```

### 5. activate env
```bash
conda activate rloed
```

### 6. run
```bash
python <script.py>
```