PROJECT_NAME = "Fresh vs Rotten Fruit Classification"

CLASS_NAMES = [
    "freshapples",
    "freshbanana",
    "freshoranges",
    "rottenapples",
    "rottenbanana",
    "rottenoranges",
]

NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

MIN_GAP_SECONDS = 60

TIME_BLOCK_DIR = "data/time_block"

TRAIN_CSV = f"{TIME_BLOCK_DIR}/train_time_block.csv"
VALIDATION_CSV = f"{TIME_BLOCK_DIR}/validation_time_block.csv"
TEST_CSV = f"{TIME_BLOCK_DIR}/test_time_block.csv"