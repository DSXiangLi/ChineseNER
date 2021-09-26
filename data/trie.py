# -*-coding:utf-8 -*-

class TreeNode(object):
    def __init__(self, char=None):
        self.char = char
        self.is_word = False
        self.id = None
        self.frequence = 0  # frequence of char
        self.children = {}

    def search(self, char):
        return self.children.get(char, None)

    def insert(self, char):
        self.children[char] = self.children.get(char, TreeNode(char=char))
        return self.children[char]


class Trie(object):
    # Faster implementation of lexicon look up compared to build_softlexicon in word_enhance
    def __init__(self):
        # empty root
        self._root = TreeNode()

    def insert(self, word, id, frequence):
        """
        Insert word into Trie along with its frequence and index
        """
        node = self._root
        for char in word:
            node = node.insert(char)
        node.is_word = True
        node.frequence = frequence
        node.id = id

    def search(self, word):
        node = self._root
        for char in word:
            node = node.search(char)
            if not node:
                return None
        return node

    def search_lexicon(self, word):
        match = self.search(word)
        if not match:
            return None
        elif not match.is_word:
            return None
        else:
            return {'lexicon': word, 'id': match.id, 'freq': match.frequence}

if __name__ == '__main__':
    tree = Trie()
    tree.insert('今天',1, 100)
    tree.insert('明天',2, 50)
    tree.search_lexicon('今天')
    tree.search_lexicon('今')
    tree.search_lexicon('明')
    tree.search_lexicon('明天')
