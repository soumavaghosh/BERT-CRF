import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL = "bert-base-uncased"
MODEL_PATH = "model.bin"
DATASET = "conll2003"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BASE_MODEL)
DEVICE = 'cuda'