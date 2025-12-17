import numpy as np
import pickle

def check():
    print("checking your embeddings...")
    # 1. load vocab and embeddings
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    embeddings = np.load("embeddings.npy")
    
    # 2. check if all embeddings are 0
    if np.all(embeddings == 0):
        print("❌ fatal error: your embeddings are all 0! training code is wrong!")
        return

    # 3. check "good" and "bad" distance
    # normal vectors, good and bad should have a significant difference, or good and great should be close
    def get_vec(word):
        return embeddings[vocab.get(word, 0)]

    v_good = get_vec("good")
    v_bad = get_vec("bad")
    v_great = get_vec("great")
    
    # calculate cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sim_good_bad = cosine_sim(v_good, v_bad)
    sim_good_great = cosine_sim(v_good, v_great)
    
    print(f"embedding dimension: {embeddings.shape}")
    print(f"Good vs Bad similarity: {sim_good_bad:.4f}")
    print(f"Good vs Great similarity: {sim_good_great:.4f}")
    
    # 4. check criteria
    if sim_good_great > sim_good_bad:
        print("✅ looks normal: Good and Great are more similar (higher similarity).")
    else:
        print("⚠️ warning: your vectors may not be trained well, Good居然觉得和 Bad 更像！")

if __name__ == "__main__":
    check()