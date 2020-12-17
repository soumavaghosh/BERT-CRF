import transformers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 20
BASE_MODEL = "bert-base-uncased"
MODEL_PATH = "model.pt"
DATASET = "conll2003"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BASE_MODEL)
DEVICE = 'cuda'