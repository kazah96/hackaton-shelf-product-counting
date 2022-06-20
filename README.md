## Usage

### Install steps
#### Deps
```
git clone https://github.com/kazah96/hackaton-shelf-product-counting.git
cd hackaton-shelf-product-counting
pip install -r requirements.txt
```
#### Models
Download models to `{project_dir}/models`
https://drive.google.com/drive/folders/1Zc6WUhM4K4CreoUSMJ_3LoHz4tnCJTIb?usp=sharing

### Run
```
py ./main.py [-v] [-o SUBMISSION_FILE] [-t TESTSET_DIR] [-m MODEL_NAME]
```
#### Params
- `-v`  Show visual result for each shelf/query
- `-o` specify filename for output submission file
- `-t` specify testset directory
- `-m` specify model name
