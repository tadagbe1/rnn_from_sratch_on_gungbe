import numpy as np

class RNN:
    def __init__(self, vocab : list, hidden_size=100, seq_length=25):
        """Initialize our RNN parameters"""
        self.hidden_size = hidden_size
        self.vocab_size = len(vocab)
        self.whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01 # (h, h)
        self.wxh = np.random.randn(self.hidden_size, self.vocab_size) * 0.01 # (h, v)
        self.why = np.random.randn(self.vocab_size, self.hidden_size) * 0.01 # (v, h)
        self.bh = np.zeros((self.hidden_size, 1))  # (h, 1)
        self.by = np.zeros((self.vocab_size, 1)) # (v, 1)
        self.seq_length = seq_length
        sorted_vocab = sorted(list(set(vocab))) # We transforme the voab into set first by precaution
        self.char_to_ix = {char: ix for ix, char in enumerate(sorted_vocab)}
        self.ix_to_char = {ix: char for char, ix in self.char_to_ix.items()}
        self.lr = 1e-1

    def encode(self, char: str):
        """Return a one hot encoding for a character we pass as input"""
        position = self.char_to_ix[char]
        x = np.zeros((self.vocab_size, 1))
        x[position, 0] = 1.0
        return x
    
    def _step(self, x, hprev):
        z = np.dot(self.whh, hprev) + np.dot(self.wxh, x) + self.bh # (h, 1)
        h = np.tanh(z) # (h, 1)
        y = np.dot(self.why, h) + self.by # (v, 1)
        p = self.softmax(y) # (v, 1)
        return h, p # return the new hidden state and the probability distribution over all the words in the vocabulary
    
    def softmax(self, x):
        e = np.exp(x - np.max(x)) # for numerical stability
        return e / np.sum(e)
    
    def loss(self, pred, target):
        """Here we implemented the cross entropy loss between a prediction and a target"""
        return -np.sum(target * np.log(pred + 1e-12)) # to avoid log(0)
    
    def forward(self, inputs, targets, hprev):
        #ys = {}
        xs, hs, ps, ys_true = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        for t in range(len(inputs)):
            x = self.encode(inputs[t])
            y_true = self.encode(targets[t])
            h, p = self._step(x, hs[t-1])
            xs[t] = x
            hs[t] = h
            ps[t] = p
            ys_true[t] = y_true
            loss +=  self.loss(p, y_true)
        return loss, xs, ys_true, hs, ps, hs[len(inputs)-1]
    
    def backward(self, xs, hs, ps, ys_true):
        # gradients initialization to zero
        d_why = np.zeros_like(self.why)
        d_by = np.zeros_like(self.by)
        d_wxh = np.zeros_like(self.wxh)
        d_whh = np.zeros_like(self.whh)
        d_bh = np.zeros_like(self.bh)
        d_h = np.zeros((self.hidden_size, 1))
        #  gradient computing at the last time step T
        T = len(xs) - 1
        delta_ty = ps[T] - ys_true[T]
        d_why += delta_ty @ hs[T].T
        d_by += delta_ty
        d_h = self.why.T @ delta_ty
        delta_t = d_h * (1 - (hs[T])**2)
        d_wxh += delta_t @ xs[T].T
        d_whh += delta_t @ hs[T-1].T
        d_bh += delta_t

        # Backprop through time for time step t<T
        for t in range(len(xs)-2, -1, -1):
            delta_ty = ps[t] - ys_true[t]
            d_why += delta_ty @ hs[t].T
            d_by += delta_ty
            d_h = self.why.T @ delta_ty + self.whh.T @ delta_t
            delta_t = d_h * (1 - (hs[t])**2)
            d_wxh += delta_t @ xs[t].T
            d_whh += delta_t @ hs[t-1].T
            d_bh += delta_t
        
        # Clip to avoid gradient explosition
        for param in [d_why, d_by, d_wxh, d_whh, d_bh]:
            np.clip(param, -5, 5, out=param)

        return d_wxh, d_whh, d_why, d_bh, d_by
    
    def sample(self, h, seed_idx, n):
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx, 0] = 1
        idxes = []
        for t in range(n):
            h, p = self._step(x, h)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())
            x = np.zeros((self.vocab_size, 1))
            x[ix, 0] = 1
            idxes.append(ix)
        return idxes
    
    def generatesample(self, h, seed_idx, n):
        idxes = self.sample(h, seed_idx, n)
        chars = [self.ix_to_char[elt] for elt in idxes]
        return ''.join(chars)


    
    def _init_adagrad_memory(self):
        m_whh = np.zeros_like(self.whh)
        m_wxh = np.zeros_like(self.wxh)
        m_why = np.zeros_like(self.why)
        m_bh = np.zeros_like(self.bh)
        m_by = np.zeros_like(self.by)

        return m_wxh, m_whh, m_why, m_bh, m_by
    
    def _update_params_adagrad(self, grads, memories):
        d_wxh, d_whh, d_why, d_bh, d_by = grads
        m_wxh, m_whh, m_why, m_bh, m_by = memories

        for param, dparam, mem in zip(
            [self.wxh, self.whh, self.why, self.bh, self.by],
            [d_wxh, d_whh, d_why, d_bh, d_by],
            [m_wxh, m_whh, m_why, m_bh, m_by]
        ):
            mem +=  dparam * dparam
            param += -self.lr * dparam / np.sqrt(mem + 1e-8)

    def prepare_sequence(self, data, p):
        inputs = [ch for ch in data[p:p + self.seq_length]]
        targets = [ch for ch in data[p+1:p + self.seq_length+1]]
        return inputs, targets
    
    def train_one_step(self, inputs, targets, hprev, memories):
        loss, xs, ys_true, hs, ps, h = self.forward(inputs, targets, hprev)
        d_wxh, d_whh, d_why, d_bh, d_by = self.backward(xs, hs, ps, ys_true)
        self._update_params_adagrad(grads=(d_wxh, d_whh, d_why, d_bh, d_by), memories=memories)
        return loss, h

    def train(self, data, threshold=5, sample_every=100):
        n, p = 0, 0
        hprev = np.zeros((self.hidden_size, 1))
        memories = self._init_adagrad_memory()
        smooth_loss = -np.log(1.0 / self.vocab_size) * self.seq_length
        while smooth_loss > threshold:
            if p + self.seq_length + 1 >= len(data) or n==0:
                hprev = np.zeros((self.hidden_size, 1))
                p = 0
            inputs, targets = self.prepare_sequence(data, p)
            loss, h = self.train_one_step(inputs, targets, hprev, memories)
            hprev = h
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            if n % sample_every == 0:
                print(f"iter {n}, loss: {smooth_loss:.4f}")
                seed_idx = self.char_to_ix[inputs[0]]
                sample_ix = self.sample(hprev, seed_idx, 200)
                txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
                print("----")
                print(txt)
                print("----")
            p += self.seq_length
            n += 1


        print("##############################################################")
        for ch, idx in self.char_to_ix.items():
            seed_idx = idx
            print(f"Seed = {seed_idx}")
            generated = self.generatesample(hprev, seed_idx, 200)
            print(generated)
            print()
        print("##############################################################\n\n")

            




    






