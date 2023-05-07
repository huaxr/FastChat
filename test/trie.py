
from transformers.tokenization_utils import Trie

if __name__ == "__main__":
    trie = Trie()
    trie.add("[CLS]")
    trie.add("extra_id_1")
    trie.add("extra_id_100")
    a = trie.split("[CLS] This is a extra_id_100")
    print(trie.data)
    print(a)