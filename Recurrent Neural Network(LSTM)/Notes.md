
# 🧠 Understanding LSTM Networks (Long Short-Term Memory)

An intuitive, step-by-step breakdown of how an **LSTM** unit manages long-term dependencies using gates and cell states.

> 📄 **Reference:** Adapted from Christopher Olah's classic blog post:
> [Understanding LSTM Networks (colah's blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---


## ⚙️ How It Works (Step-by-Step)

```
[Previous State C_{t-1}] ----> ( x ) -----------> ( + ) ------> [New State C_t]
                                 |                 |
                         (Forget Gate f_t)  (Input i_t * C~_t)
                                 ^                 ^
                                 |                 |
[Input x_t, Hidden h_{t-1}] -----+-----------------+------------> (Output Gate o_t) ---> [New Hidden h_t]

```

### 1. The Forget Gate Layer (What to Erase)

The first step decides what information to throw away from the previous cell state ($C_{t-1}$).

* **Mechanism:** A **sigmoid** layer takes $h_{t-1}$ and $x_t$, outputting a value between **$0$** (*completely forget*) and **$1$** (*completely keep*) for each number in $C_{t-1}$.
* 💬 **Language Model Example:** Erasing the gender of an *old* subject when a new subject is introduced in a text.

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

---

### 2. The Input Gate Layer (What to Add)

The second step determines what new information to store in the cell state. This happens in two sub-steps:

1. **Input Gate ($\sigma$):** Decides *which* values to update ($i_t$).
2. **Candidate Layer ($\tanh$):** Generates a vector of new candidate values ($\tilde{C}_t$) bounded between $-1$ and $1$.

* 💬 **Language Model Example:** Preparing to add the gender of the *new* subject to replace the forgotten one.

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

---

### 3. Updating the Cell State

Now, the old state ($C_{t-1}$) is updated into the new state ($C_t$):

1. **Multiply** $C_{t-1}$ by $f_t$ to drop information marked for deletion.
2. **Add** $i_t * \tilde{C}_t$ to include the scaled candidate values.

* 💬 **Language Model Example:** Actually dropping the old gender flag and storing the new gender flag.

$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

---

### 4. The Output Gate Layer (What to Pass Along)

Finally, the model decides what to output as the new hidden state ($h_t$):

1. A **sigmoid layer** decides which parts of the cell state to output ($o_t$).
2. The updated cell state ($C_t$) passes through a **$\tanh$** layer to push values to $[-1, 1]$.
3. Multiply the two outputs so only selected information is emitted.

* 💬 **Language Model Example:** If the model just saw a subject, it might output grammatical tense or number (singular vs. plural) to help conjugate the next verb.

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t * \tanh(C_t)$$

---

> **💡 Key Takeaway:** The **cell state** acts as a conveyor belt, while the **forget, input, and output gates** act as precision control valves that regulate what is retained, updated, and transmitted.

---

## 🔗 Resources & Further Reading

* 📖 **Original Article:** [Understanding LSTM Networks by Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
