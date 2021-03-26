# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf
from numpy import pi, sqrt


# %%
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)
physical_devices

# %% [markdown]
# ## Position and Embedding

# %%
class PositionEmbeding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(PositionEmbeding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)
        self.start = 1
    
    def get_token_embedding(self):
        return self.token_emb.weights
    
    def set_start(self, N):
        #self.start = N
        i = 1
    
    def position_embedding(self, x):
        batch_length = tf.shape(x)[1] 
        batch_size = tf.shape(x)[0]
        
        pos = tf.reshape(tf.tile(tf.range(self.start, batch_length + self.start), [batch_size]),
                                       [batch_size, batch_length])
        pos = tf.cast(pos, tf.int32)
        invalid_pos = tf.cast(tf.not_equal(pos, 0), tf.int32)
        pos *= invalid_pos
        
        return self.pos_emb(pos)
        
    def token_embedding(self, x):
        words = self.token_emb(x)
        return words
    
    def call(self, input):
        positions = self.position_embedding(input)
        words = self.token_embedding(input)
        return words + positions 


# %%
tf.range(1, 10 + 1)

# %% [markdown]
# ## GELU Function

# %%
def GELU(x):
    return 0.5*x*(1+tf.tanh(sqrt(2/pi)*(x+0.044715*tf.pow(x, 3))))

# %% [markdown]
# ## Multi Head Attention

# %%
mierda = 0
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
    
    
    def split_heads(self, x):
        global mierda
        *prev, embedding_dim = x.shape.as_list()
        mierda = x
        x = tf.reshape(x, [-1] + prev[1:] + [self.num_heads, embedding_dim//self.num_heads])
        return tf.transpose(x, [0, 2, 1, 3])
    
    
    '''def apply_mask(self, w):
        matrix_dim = w.shape[-1]
        mask = create_look_ahead_mask(matrix_dim)
        mask = tf.reshape(mask, [1,1,matrix_dim,matrix_dim])
        return w*mask - tf.cast(1e10, w.dtype)*(1-mask)'''
    
    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        *prev, heads, sub_dim = x.shape.as_list()
        x = tf.reshape(x, [-1] + prev[1:] + [heads*sub_dim])
        return x
    
    def call(self, input):
        
        input, past, mask = input
        
        #Para la dimension 3, que ahora mismo tenemos [batch, seq_len, embedding*3]
        # dividimos en 3 partes para generar query, key y value que al mismo tiempo
        # dividimos en los heads que queramos
        q, k, v = map(self.split_heads, tf.split(input, 3, axis=2))
        
        
        present = tf.stack([k,v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk,k], axis=-2)
            v = tf.concat([pv,v], axis=-2)
        
        #Por las siguientes multiplicaciones tendremos que w tiene dimensiones
        #[batch, heads, dst_sec, orig_sec] es decir tenemos para todos los elementos del batch
        #y para todos los heads una matriz donde cada columna y fila son palabras, las diagonales
        #son las mismas palabras y el valor es la "puntuación" o relación que esta palabra tiene
        #con la otra
        w = tf.matmul(q, k, transpose_b=True)
        w /= tf.math.rsqrt(tf.cast(v.shape.as_list()[-1], w.dtype))
        
        if mask is not None:
            w += (mask * -1e9)
        
        #Aplicamos la mascara, softmax para regular entre 0 y 1 y finalmente multiplicamos los valores
        #de query y key por value
        #w = self.apply_mask(w)
        w = tf.nn.softmax(w, axis=-1)
        output = tf.matmul(w, v)
        
        return self.merge_heads(output), present
        

# %% [markdown]
# ## Decoder Block

# %%
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(DecoderBlock, self).__init__()
        self.MultiHeadAttention = MultiHeadAttention(num_heads=num_heads)
        
        self.LayerNorm1 = tf.keras.layers.LayerNormalization()
        self.LayerNorm2 = tf.keras.layers.LayerNormalization()
        
        self.GELU = tf.keras.layers.Activation(GELU)
        
        self.conv1D_1 = tf.keras.layers.Conv1D(d_model*3, 1)
        self.conv1D_2 = tf.keras.layers.Conv1D(d_model*4, 1)
        
        self.conv1D_3 = tf.keras.layers.Conv1D(d_model, 1)
        self.conv1D_4 = tf.keras.layers.Conv1D(d_model, 1)

    
    def call(self, input):
        x, past, mask = input
        x_norm1 = self.LayerNorm1(x)
        x_conv1 = self.conv1D_1(x_norm1)
        x_attn, present = self.MultiHeadAttention((x_conv1, past, mask))
        x_conv3 = self.conv1D_3(x_attn)
        
        x = tf.keras.layers.Add()([x_conv3,x])
        
        x_norm2 = self.LayerNorm2(x)
        x_conv2 = self.conv1D_2(x_norm2)
        
        #https://mlfromscratch.com/activation-functions-explained/#/
        x_gelu = self.GELU(x_conv2)
        
        # Hay paginas que hablan de dos densas, pero en el codigo se utiliza otra cosa WTF??
        # https://www.reddit.com/r/MachineLearning/comments/b1c6sn/d_is_gpt2_source_code_publically_available/eilbqas/
        # en ese enlace se menciona posible reduccion del codigo?
        
        # Si ponemos dos sensas no llegamos a lo pedido (en cuanto a pesos) indicados en el paper
        # entonces es menos potente?? O estan contando pesos fuera del model, pero no tiene sentido
        
        
        
        x_conv4 = self.conv1D_4(x_gelu)
        
        x = tf.keras.layers.Add()([x_conv4,x])
        return x, present
        

# %% [markdown]
# ## GPT2

# %%
class GPT2(tf.keras.Model):
    def __init__(self, num_layers, num_heads, d_model, vocab_size, max_len):
        super(GPT2, self).__init__()
        self.DecoderBlocks = [DecoderBlock(num_heads, d_model) for _ in range(num_layers)]
        self.PosEmb = PositionEmbeding(max_len, vocab_size, d_model)
        self.Norm = tf.keras.layers.LayerNormalization()
        self.vocab = vocab_size
        self.d_model = d_model
        self.pasts = list([None] * num_layers)
        self.conditioned = False
        
    def initialize_past_status(self, initial_input):
        self.PosEmb.set_start(len(initial_input))
        logit_past, pasts = self.call(initial_input)
        self.pasts = pasts
        self.conditioned = True
        return logit_past
    
    def reset_past_status(self):
        self.pasts = [None] * num_layers
        self.conditioned = False
    
    def generate_next_output(self, X):
        
        output, present = self.call(X)
        
        if self.conditioned:
            self.pasts = present
        else:
            self.pasts = tf.concat([self.pasts, present], axis=-2)
        
        return output

    
    def create_look_ahead_mask(self, size):
        mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask
    
    def create_paddding_mask(self, inp):
        padded = tf.cast(tf.math.equal(inp, 0), tf.float32)
        return padded[:, tf.newaxis, tf.newaxis, :]
    
    def create_mask(self, x):
        matrix_dim = tf.shape(x)[-1]
        
        mask_ahead = self.create_look_ahead_mask(matrix_dim)
        mask_padd = self.create_paddding_mask(x)
        
        mask = tf.maximum(mask_padd, mask_ahead)
        return mask
    
    def call(self, input):
        presents = []
        
        mask = self.create_mask(input)
        x = self.PosEmb(input)
        
        if self.conditioned:
            pasts_unstack = tf.unstack(self.pasts, axis=1)
        else:
            pasts_unstack = self.pasts
        
        for DecoderB, past in enumerate(pasts_unstack):
            x, present = self.DecoderBlocks[DecoderB]((x, past, mask))
            presents.append(present)
            
        x = self.Norm(x)
        shape = tf.shape(x)
        x = tf.reshape(x, [shape[0] * shape[1], self.d_model])
        x = tf.matmul(x, self.PosEmb.get_token_embedding(), transpose_b=True)
        logits = tf.reshape(x, [-1, shape[1], self.vocab])
        return logits, tf.stack(presents, axis=1)
    
    def train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.int32)

        with tf.GradientTape() as tape:
            y_pred, _ = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
  
        y_pred, _ = self(x, training=False)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}
        

# %% [markdown]
# # Generate Dataset

# %%
import os
Root = os.getcwd()


# %%
path_originalData = Root + "/Dataset"
all_files = [os.path.join(path_originalData,file) for file in os.listdir(path_originalData)]

# %% [markdown]
# #### Fix posible problems in Unicode Files

# %%
from ftfy import fix_text


# %%
procesed_path = Root + "/Processed.txt"
writer = open(procesed_path, "w")
for file in all_files:
    f = open(file, "r")
    writer.writelines([fix_text(line, normalization="NFKC") for line in f.readlines()])
    f.close()
writer.close()

# %% [markdown]
# #### Byte Pair Encoding

# %%
from collections import Counter
import sentencepiece as spm
import numpy as np
import csv


# %%
token_count = Counter()
with open(procesed_path, 'r') as f:
    for line in f:
        token_count.update(line.lower().split())


# %%
counter_path = Root + "/Counter.txt"
with open(counter_path, 'w', newline='') as f:
    output = csv.writer(f, delimiter='\t')
    for word in token_count:
        output.writerow([word, token_count[word]])


# %%
#Libreria que nos crea la codificación byte Pair Encoding (el vocab_size es a nuestra elección)
Model_path = Root + "/BPE_Model"
vocab_size = int(len(token_count)/2)

spmcmd = '--input={spm_input} --model_prefix={spm_model} --input_format=tsv --vocab_size={vocab_size} --user_defined_symbols=[SEP],[BOS],[EOS] --hard_vocab_limit=false --model_type=bpe --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK]'.format(
spm_input=counter_path, spm_model=Model_path, vocab_size=vocab_size)
spm.SentencePieceTrainer.train(spmcmd)


# %%
s = spm.SentencePieceProcessor()
s.Load(Model_path + ".model")


# %%
BOS = 3
EOS = 4


# %%
dataset = []

count = 0
with open(counter_path, 'r') as f:
    for line in f:
        encod = s.encode_as_ids(line)

        dataset += [[[BOS]+ encod, encod + [EOS]]]
            

# %% [markdown]
# # Load Dataset

# %%
train_percent = (85 / 100)


# %%
from sklearn.model_selection import train_test_split


# %%
X = [X[0] for X in dataset]
Y = [Y[1] for Y in dataset]


# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1-train_percent)


# %%
max_len = max(max([len(x) for x in X_train]), max([len(x) for x in X_test]))


# %%
for i in range(len(X_train)):
    X_train[i] += [0 for _ in range(max_len-len(X_train[i]))]
    y_train[i] += [0 for _ in range(max_len-len(y_train[i]))]


# %%
for i in range(len(X_test)):
    X_test[i] += [0 for _ in range(max_len-len(X_test[i]))]
    y_test[i] += [0 for _ in range(max_len-len(y_test[i]))]


# %%
BATCH_SIZE = 128
EPOCHS = 10
BUFFER_SIZE = 10000


# %%
train_dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X_train), tf.constant(y_train))).batch(BATCH_SIZE, drop_remainder=True)
test_dataset  = tf.data.Dataset.from_tensor_slices((tf.constant(X_test), tf.constant(y_test))).batch(BATCH_SIZE, drop_remainder=True)

# %% [markdown]
# # Training

# %%
num_layers = 12
num_heads = 4
d_model = 768


# %%
gpt2 = GPT2(num_layers, num_heads, d_model, vocab_size, max_len)


# %%
def Loss(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)
    
    mask = tf.cast(mask, loss.dtype)
    loss *= mask
    
    loss = tf.reduce_sum(loss, axis=1)
    average_loss = tf.reduce_mean(loss / tf.reduce_sum(mask, axis=1))
    
    
    
    return tf.reduce_mean(loss)  


# %%
gpt2.compile(optimizer='adam', loss=Loss, metrics=['accuracy'])


# %%
hist = gpt2.fit(train_dataset, validation_data=test_dataset)


# %%
def top_k_logits(logits, k):
    if k == 0:
        return logits

    values, _ = tf.nn.top_k(logits, k=k)
    min_values = values[:, -1]

    return tf.where(
        logits < min_values,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )


# %%
def extract_data(logits, temperature=0.7, top_k=10):
    logits = logits[:, -1, :]/tf.cast(temperature, tf.float32)
    logits = top_k_logits(logits, k=top_k)
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32)
    return sample


# %%
def generator_text(initial_sentence, model, seq_len):
    if initial_sentence == None:
        print("Intial Sentence Required")
    else:
        gpt2.reset_past_status()
        context = tf.expand_dims(([BOS] + s.encode_as_ids(initial_sentence)), 0)
        act_logits = model.initialize_past_status(context)
        sample = extract_data(act_logits)
        all_output = tf.concat([context, sample], axis=-1)
        next_input = sample
        for i in range(seq_len):
            logits = model.generate_next_output(next_input)
            sample = extract_data(logits)
            
            if tf.equal(sample, EOS):
                break
            
            all_output = tf.concat([context, sample], axis=-1)
            next_input = sample
        
        result = tf.squeeze(all_output, axis=0)
        pred = [int(i) for i in result]
        generated_seq = s.decode_ids(pred[1:])
        generated_seq = generated_seq.replace("[SEP]", "").strip()
        generated_seq = ' '.join(initial_sentence.split() + generated_seq.split())
        return generated_seq


# %%
Initial_Sentence = "Hello world today is"


# %%
generator_text(Initial_Sentence, gpt2, 250)


