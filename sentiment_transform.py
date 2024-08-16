import tensorflow as tf

LABEL_KEY = "sentiment"
FEATURE_KEY = "review"
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def cleaningText(text):
    text = tf.strings.regex_replace(text, r'@[A-Za-z0-9]+', '')
    text = tf.strings.regex_replace(text, r'#[A-Za-z0-9]+', '')
    text = tf.strings.regex_replace(text, r'RT[\s]', '')
    text = tf.strings.regex_replace(text, r"http\S+", '')
    text = tf.strings.regex_replace(text, r'[0-9]+', '')
    text = tf.strings.regex_replace(text, r'[^\w\s]', '')
    text = tf.strings.regex_replace(text, r'\n', ' ')
    text = tf.strings.regex_replace(text, r'\s+', ' ')
    text = tf.strings.strip(text)

    return text

def casefoldingText(text):
    text = tf.strings.lower(text)
    return text

def transform_text(text):
    text = cleaningText(text)
    text = casefoldingText(text)
    return text

def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs: map from feature keys to raw features.

    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    outputs[transformed_name(FEATURE_KEY)] = transform_text(inputs[FEATURE_KEY])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
